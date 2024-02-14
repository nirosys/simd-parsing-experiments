#include <iostream>
#include <fstream>
#include <span>
#include <immintrin.h>
#include <cassert>
#include <stdio.h>
#include <unistd.h>

#include <cxxopts.hpp>
#include <fmt/core.h>

namespace arch::avx {
   struct Input {
      typedef __m256i SIMDType;
      __m256i chunk_1;

      void load(const std::span<uint8_t> &data) {
         uint8_t buffer[32] = {0};
         assert(data.size_bytes() <= 32);
         // TODO: Handle data > 32 bytes
         memcpy(buffer, data.data(), data.size_bytes());

         chunk_1 = _mm256_loadu_si256((const SIMDType *)buffer);
      }

      uint32_t find_structurals() {
         __m256 opcode = _mm256_set1_epi8(0xFF);
         __m256 opcodes_cmp = _mm256_cmpeq_epi8(chunk_1, opcode);

         uint32_t structurals = (uint32_t)_mm256_movemask_epi8(opcodes_cmp);

         return structurals;
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

template <class Input>
class Deserializer {
   Input simd_input;
   std::span<uint8_t> data;

   // This tape differs from the commanum example in that, rather than having to track when we
   // see a digit, or a comma, we're instead going to just track when we see a new number, and
   // it's length based on the next number.
   auto build_tape(uint32_t structurals) -> std::vector<std::span<uint8_t>> {
      std::vector<uint8_t> offsets = simd_input.flatten_structurals(structurals);
      std::vector<std::span<uint8_t>> spans;

      size_t last_offset = 0;
      for (auto it = offsets.begin(); it != offsets.end(); it++) {
         // our span is from where we are, to our next offset..
         auto it_next = std::next(it, 1);
         if (it_next == offsets.end()) {
            spans.push_back(data.subspan(*it));
         } else {
            spans.push_back(data.subspan(*it, *it_next - *it));
         }
      }

      return spans;
   }

   auto parse_number(const std::span<uint8_t> num_bytes) -> std::variant<uint32_t, std::error_code> {
      assert(num_bytes.size_bytes() <= 5);

      uint64_t encoded = 0x0;
      memcpy(&encoded, num_bytes.data(), num_bytes.size_bytes());
      uint32_t num = _pext_u64(encoded, 0x7f7f7f7f7f7f);
      return (uint32_t)num;
   }

   public:

   Deserializer(std::vector<uint8_t> &given) {
      data = std::span { given };
      simd_input.load(data.subspan(0, 32)); // TODO: support more than 32-bytes
   }

   auto parse_nums() -> std::vector<uint32_t> {
      std::vector<uint32_t> nums;
      uint32_t structurals = simd_input.find_structurals();
      auto tape = build_tape(structurals);

      for (auto span : tape) {
         auto num = parse_number(span.subspan(1));
         if (std::holds_alternative<std::error_code>(num)) {
            std::cout << "Error: unexpected data while parsing number." << std::endl; // TODO: Need to map spans to offset.
            break;
         }
         nums.push_back(std::get<uint32_t>(num));
      }
      return nums;
   }

};


auto encode_int(uint32_t n) -> std::vector<uint8_t> {
   std::vector<uint8_t> ret;

   // we have 4 bytes.. which means we have (at most) 5 bytes.
   int bits_needed = 32 - _lzcnt_u32(n);
   int bytes_needed = std::max((bits_needed / 7) + (bits_needed % 7)/(bits_needed % 7), 1);
   ret.reserve(bytes_needed + 1);

   ret.push_back(0xFF); // Our integer marker.
   for (int i=0; i < bytes_needed; i++) {
      ret.push_back(n & 0x7f);
      n = n >> 7;
   }

   return ret;
}

typedef std::vector<uint8_t> Bytes;

Bytes read_all(std::istream &in) {
   Bytes all_bytes;
   all_bytes.resize(32); // We only support at most 32bytes currently..
   in.read(reinterpret_cast<char*>(all_bytes.data()), 32);
   all_bytes.resize(in.gcount()); // truncate to the size of our actual data..
   return all_bytes;
}


int main(int argc, char **argv) {
   cxxopts::Options options("binarynums", "An experiment in SIMD parsing of binary data");
   options.add_options()
      ("e,encode", "Encode a comma separated list of numbers into binary", cxxopts::value<std::vector<uint32_t>>())
      ("d,decode", "Decode a file (or '-' for STDIN) from binary.", cxxopts::value<std::string>()->default_value("-"))
      ("b,benchmark", "Run benchmarks.", cxxopts::value<bool>()->default_value("false"))
      ("r,raw", "Output in raw binary (otherwise ASCII encode)", cxxopts::value<bool>()->default_value("false"))
      ("INPUT", "Input data", cxxopts::value<std::vector<uint32_t>>())
      ("h,help", "Print usage");

   options.parse_positional({"INPUT"});

   auto result = options.parse(argc, argv);

   if (result.count("help")) {
      std::cout << options.help() << std::endl;
      return 0;
   }

   if (result["encode"].count()) {
      auto nums = result["encode"].as<std::vector<uint32_t>>();
      for (auto i : nums) {
         auto bytes = encode_int(i);
         if (!result["raw"].as<bool>()) {
            for (auto b : bytes) {
               std::cout << fmt::format("{:#x} ", b);
            }
            std::cout << std::endl;
         } else {
            for (auto b : bytes) {
               std::cout << b;
            }
         }
      }
   }

   if (result["decode"].count()) {
      Bytes data;
      std::string filename = result["decode"].as<std::string>();
      if (filename != "-") {
         std::ifstream ifile(filename);
         if (ifile) {
            data = read_all(ifile);
         } else {
            std::cerr << "Error opening file." << std::endl;
            return -1;
         }
      } else {
         data = read_all(std::cin);
      }

      if (data.size() > 32) {
         std::cerr << "Current implementation does not support more than 32 bytes of input." << std::endl;
      }

      Deserializer<arch::avx::Input> deser(data);
      auto nums = deser.parse_nums();
      std::cout << "Numbers:" << std::endl;
      for (auto n : nums) {
         std::cout << "   " << n << std::endl;
      }
   }

   return 0;
}
