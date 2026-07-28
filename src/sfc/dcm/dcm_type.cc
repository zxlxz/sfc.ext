#include "sfc/dcm/dcm_type.h"

namespace sfc::dcm {

auto DcmVR::is_int() const -> bool {
  static const DcmVR v[] = {US, UL, UV, SS, SL, SV};
  return Slice{v}.contains(*this);
}

auto DcmVR::is_flt() const -> bool {
  static const DcmVR v[] = {FL, FD};
  return Slice{v}.contains(*this);
}

auto DcmVR::is_str() const -> bool {
  static const DcmVR v[] = {SH, LO, UC, AS, AE, CS, PN, UI, UR, ST, LT, UT, TM, IS, DS, DA, DT};
  return Slice{v}.contains(*this);
}

auto DcmVR::is_buf() const -> bool {
  static const DcmVR v[] = {OB, OW, OL, OV, OF, OD, UN};
  return Slice{v}.contains(*this);
}

auto DcmVR::use_len32() const -> bool {
  static const DcmVR v[] = {OB, OW, OL, OV, OF, OD, UN};
  return Slice{v}.contains(*this);
}

template <>
auto DcmTVR::PixelData<u8>() -> DcmTVR {
  return {{0x7FE0, 0x0010}, OB};
}

template <>
auto DcmTVR::PixelData<u16>() -> DcmTVR {
  return {{0x7FE0, 0x0010}, OW};
}

template <>
auto DcmTVR::PixelData<f32>() -> DcmTVR {
  return {{0x7FE0, 0x0008}, OF};
}

}  // namespace sfc::dcm
