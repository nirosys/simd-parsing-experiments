#include <bitset>
#include <iostream>
#include <string_view>
#include <vector>
#include <immintrin.h>

// A simple enum to denote the types of tokens we're tracking.
enum class TokenType {
   Unknown,
   Integer,
   Comma,
};

// The type, and starting offset of a token.
struct Token {
   TokenType tpe;
   size_t offset;
};

// Namespace to separate out the platform specific implementation. If we wanted to support more than just AVX we could
// include separate namespaces, or just new types that can be used as template arguments for the deserializers.
namespace arch::avx {
   // Input type contains the input, as it is used by SIMD implementations.  Here we store the actual
   // SIMD register(s) containing the chunk of data we're working on, and provide any functionality for
   // managing and interpreting it.
   struct Input {
      typedef __m256i SIMDType;
      __m256i chunk;

      // Load a new chunk of data into the vector(s) we're using.
      void load(const std::string_view &data) {
         char buffer[32] = {0x20}; // buffer so we can ensure the data is padded.
         memcpy(buffer, data.data(), data.length());
         chunk = _mm256_loadu_si256((const SIMDType *)buffer);
      }

      // Find all of the structurals and, in this case, pseudo-structurals contained within the data.
      // A structural, would be any character that indicates physical boundaries. This data representation
      // only has one and that is the comma.
      //
      // A pseudo-structural, would be any character that is not a structural and is not whitespace, but
      // follows a structural or whitespace. In this case, a digit would be a pseudo structural.
      //
      // Since we're indexing commas, and anything that follows whitespace or commas, it's left up to the
      // materialization of the data to validate invalid numeric "columns" like "12F5". If we wanted to
      // identify these things sooner, we could include digits here, and track the first digit in a span
      // of digits. That would require a bit more than we have here so I've deferred it to materialization.
      uint32_t find_structurals() {
         // First, mask out our spaces.
         __m256 spaces = _mm256_set1_epi8(0x20);
         __m256 spaces_cmp = _mm256_cmpeq_epi8(chunk, spaces);

         // Second, mask out our commas.
         __m256 commas = _mm256_set1_epi8(0x2C);
         __m256 commas_cmp = _mm256_cmpeq_epi8(chunk, commas);

         // we need commas + (whitespace >> 1 & ~whitespace)
         uint32_t ws_mask = (uint32_t)_mm256_movemask_epi8(spaces_cmp);
         uint32_t comma_mask = (uint32_t)_mm256_movemask_epi8(commas_cmp);
         uint32_t pseudo_structurals = ~(comma_mask | ws_mask) // Not commas or whitespace..
            & (((comma_mask | ws_mask) << 1) | 1);             // but come after a comma, whitespace, or nothing.

         uint32_t poi_mask = pseudo_structurals | comma_mask; // PoIs are structurals & pseudo-structurals.

         return poi_mask;
      }

      // Flattens the structural bitmask into a list of offsets that we can enumerate to find the structural,
      // and pseudo-structural points of interest.
      std::vector<uint8_t> flatten_structurals(uint32_t structurals) {
         std::vector<uint8_t> offsets;
         size_t n = _mm_popcnt_u32(structurals);
         offsets.reserve(((n / 8) + 1) * 8);

         // We don't really have to unroll this loop for this trivial implementation, but
         // if we were to expand to larger datasets rather than just this 32byte PoC, we'd
         // want to quickly generate the list of offsets, and unwinding this into 8 32bit words,
         // then adding the chunk's offset to them via a vector operation would be more efficient.
         // This gets us a little of the way there, and allows us to add (in more bulk) the values
         // to our vector.
         while (structurals != 0) {
            uint8_t v0 = _mm_tzcnt_32(structurals);
            structurals &= structurals - 1;
            uint8_t v1 = _mm_tzcnt_32(structurals);
            structurals &= structurals - 1;
            uint8_t v2 = _mm_tzcnt_32(structurals);
            structurals &= structurals - 1;
            uint8_t v3 = _mm_tzcnt_32(structurals);
            structurals &= structurals - 1;
            uint8_t v4 = _mm_tzcnt_32(structurals);
            structurals &= structurals - 1;
            uint8_t v5 = _mm_tzcnt_32(structurals);
            structurals &= structurals - 1;
            uint8_t v6 = _mm_tzcnt_32(structurals);
            structurals &= structurals - 1;
            uint8_t v7 = _mm_tzcnt_32(structurals);
            structurals &= structurals - 1;
            offsets.insert(offsets.end(), { v0, v1, v2, v3, v4, v5, v6, v7 });
         }
         offsets.resize(n);

         return offsets;
      }
   };
}

// This type wraps the provided input along with a 256bit vector register used for tokenization, and offers functionality
// for parsing the comma separated integers.
template <class Input>
class Deserializer {
   Input simd_input;
   std::string_view str; // Our data is coming from our environment (commandline args) and will exist the duration of our process.

   auto build_tape(uint32_t structurals) -> std::vector<Token> {
      std::vector<uint8_t> offsets = simd_input.flatten_structurals(structurals);
      std::vector<Token> tokens;
      int state = 0; // 0 = num, 1 = comma, 3 = err

      for (auto offset : offsets) {
         auto c = str.at(offset);
         if ((state == 0) && ::isdigit(c)) {
            tokens.push_back(Token { .tpe = TokenType::Integer, .offset = offset });
            state = 1;
         } else if ((state == 1) && (c == ',')) {
            tokens.push_back(Token { .tpe = TokenType::Comma, .offset = offset });
            state = 0;
         } else {
            std::cout << "Error: unexpected character: '" << c << "' at offset " << offset << std::endl;
            break;
         }
      }
      return tokens;
   }


   // Simple function to materialize a sequence of digits into a 32bit integer.
   // Since validation of numeric sequences was deferred to materialization, we need to ensure that our digits
   // stop at a space, or comma.
   auto parse_number(const std::string_view &num) -> std::variant<int, std::error_code> {
      int converted = 0;
      int i = 0;
      while ((i < num.length()) && ::isdigit(num.at(i))) {
         converted *= 10; // We have another digit, so scale our accumulator by 10
         converted += num.at(i) ^ 0x30; // we already tested that this is a digit.. high nibble will be 0x3
         i++;
      }

      if (i < num.length()) { // we have more characters, so make sure we're ending on a space or comma.
         auto c = num.at(i);
         if ( (c != ' ') && (c != ',') ) {
            return std::error_code(1, std::generic_category());
         }
      }
      return converted;
   }

   public:

   Deserializer(const std::string_view &input_str) {
      simd_input.load(input_str);
      str = input_str;
   }

   // Primary entry point for parsing the numbers in a comma delimited text.
   auto parse_nums() -> std::vector<int> {
      std::vector<int> nums;
      uint32_t structurals = simd_input.find_structurals();
      std::vector<Token> tape = build_tape(structurals); // Generate our Tape, for value walking.

      for (auto token : tape) {
         switch (token.tpe) {
            case TokenType::Integer: {
               std::string_view view(str.data() + token.offset, str.length() - token.offset);
               auto n = parse_number(view);
               if (std::holds_alternative<std::error_code>(n)) {
                  std::cout << "ERROR: Unexpected character while parsing number around offset " << token.offset << std::endl;
                  break;
               }
               nums.push_back(std::get<int>(n));
               break;
            }
            case TokenType::Comma:
               break;
            default:
               break;
         }
      }

      return nums;
   }
};

// Main entry point, implement our CLI..
int main(int argc, char **argv) {
   if (argc < 2) {
      std::cout << "usage: " << argv[0] << " <delimited numbers>" << std::endl;
      return 1;
   }

   std::string_view input_str(argv[1]);
   if (input_str.length() > 32) {
      std::cout << "This PoC currently only supports text up to 32 characters long. Sorry." << std::endl;
      return 1;
   }

   Deserializer<arch::avx::Input> input(input_str);
   auto nums = input.parse_nums();

   std::cout << "Numbers:" << std::endl;
   for (auto n : nums) {
      std::cout << "   " << n << std::endl;
   }
   return 0;
}
