#include "nct/dcm/dcm_type.h"

namespace nct::dcm {

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

}  // namespace nct::dcm
