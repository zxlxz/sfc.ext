#include "nct/dcm/dcm_elmt.h"

namespace nct::dcm {

DcmElmt::DcmElmt(DcmTVR tvr, DcmVar var) noexcept : _tvr{tvr}, _var{mem::move(var)} {}

DcmElmt::~DcmElmt() {}

DcmElmt::DcmElmt(DcmElmt&& other) noexcept = default;

DcmElmt& DcmElmt::operator=(DcmElmt&& other) noexcept = default;

auto DcmElmt::from_int(DcmTVR tvr, i64 val) -> DcmElmt {
  return DcmElmt{tvr, DcmVar{val}};
}

auto DcmElmt::from_flt(DcmTVR tvr, f64 val) -> DcmElmt {
  return DcmElmt{tvr, DcmVar{val}};
}

auto DcmElmt::from_str(DcmTVR tvr, String val) -> DcmElmt {
  return DcmElmt{tvr, DcmVar{mem::move(val)}};
}

auto DcmElmt::from_buf(DcmTVR tvr, List<u8> buf) -> DcmElmt {
  return DcmElmt{tvr, DcmVar{mem::move(buf)}};
}

auto DcmElmt::as_int() const -> i64 {
  return _var.as<i64>().unwrap_or(0);
}

auto DcmElmt::as_flt() const -> f64 {
  return _var.as<f64>().unwrap_or(0.0);
}

auto DcmElmt::as_str() const -> Str {
  return _var.as<String>().unwrap_or({}).as_str();
}

auto DcmElmt::as_buf() const -> Slice<const u8> {
  auto to_slice = [](auto& v) { return v.as_slice(); };
  return _var.as<List<u8>>().map(to_slice).unwrap_or({});
}

auto DcmElmt::as_mut_buf() -> Slice<u8> {
  auto to_slice = [](auto& v) { return v.as_mut_slice(); };
  return _var.as_mut<List<u8>>().map(to_slice).unwrap_or({});
}

auto DcmElmt::serialize_size() const -> usize {
  const auto buf = this->as_buf();
  const auto buf_len = num::align_up(buf.len(), usize{2U});

  const auto use_len32 = _tvr.vr.use_len32();
  const auto len_field_size = use_len32 ? 6U : 2U;

  return 4U + 2U + len_field_size + buf_len;
}

}  // namespace nct::dcm
