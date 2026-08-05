#pragma once

#include "sfc/alloc.h"
#include "nct/dcm/dcm_type.h"

namespace nct::dcm {

using DcmLen = u32;
using DcmVar = sfc::Variant<i64, f64, String, List<u8>>;

class DcmElmt {
  DcmTVR _tvr;
  DcmVar _var;

 public:
  DcmElmt(DcmTVR tvr, DcmVar data) noexcept;
  ~DcmElmt();

  DcmElmt(DcmElmt&& other) noexcept;
  DcmElmt& operator=(DcmElmt&& other) noexcept;

 public:
  static auto from_int(DcmTVR tvr, i64 val) -> DcmElmt;
  static auto from_flt(DcmTVR tvr, f64 val) -> DcmElmt;
  static auto from_str(DcmTVR tvr, String val) -> DcmElmt;
  static auto from_buf(DcmTVR tvr, List<u8> buf) -> DcmElmt;

  auto as_int() const -> i64;
  auto as_flt() const -> f64;
  auto as_str() const -> Str;

  auto as_buf() const -> Slice<const u8>;
  auto as_mut_buf() -> Slice<u8>;

  auto serialize_size() const -> usize;

  auto tag() const noexcept -> DcmTag {
    return _tvr.tag;
  }

  auto vr() const noexcept -> DcmVR {
    return _tvr.vr;
  }
};

}  // namespace nct::dcm
