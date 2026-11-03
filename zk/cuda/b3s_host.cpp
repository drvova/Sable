#include "b3s.h"

extern "C" {
uint64_t b3s_size() { return sizeof(B3S); }
void b3s_init_c(B3S* s) { b3s_init(s); }
void b3s_update_c(B3S* s, const uint8_t* d, uint64_t n) { b3s_update(s, d, n); }
void b3s_digest_c(const B3S* s, uint8_t* out, uint64_t n) { b3s_digest(s, out, n); }
}
