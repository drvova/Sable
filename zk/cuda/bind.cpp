#include <torch/extension.h>

void set_tw_offsets(torch::Tensor off);
void encode(torch::Tensor msg, torch::Tensor out, torch::Tensor tw, int64_t n, int64_t level);
void encode_sel(torch::Tensor msg, torch::Tensor out, torch::Tensor tw, int64_t n, int64_t level, torch::Tensor pos);
void leaf_hash(torch::Tensor cw, torch::Tensor out);
void node_hash(torch::Tensor in, torch::Tensor out);
void ext_mul(torch::Tensor a, torch::Tensor b, torch::Tensor c);
int64_t grind(torch::Tensor base, int64_t bits);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("set_tw_offsets", &set_tw_offsets);
  m.def("encode", &encode);
  m.def("encode_sel", &encode_sel);
  m.def("leaf_hash", &leaf_hash);
  m.def("node_hash", &node_hash);
  m.def("ext_mul", &ext_mul);
  m.def("grind", &grind);
}
