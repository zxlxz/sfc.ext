#pragma once

#include "sfc/fs.h"
#include "sfc/dcm/dcm_elmt.h"

namespace sfc::math {
template <class T, u32 N>
class NdArray;
}

namespace sfc::dcm {

struct DcmMeta {
  Str SOPClassUID = "1.2.840.10008.5.1.4.1.1.2";  // CT
  Str TransferSyntaxUID = "1.2.840.10008.1.2.1";  // Little Endian
};

class DcmFile {
  static constexpr u32 kHeadSize = 128U;
  fs::File _file;

 public:
  static auto open(Str path) -> io::Result<DcmFile>;
  static auto create(Str path) -> io::Result<DcmFile>;

 public:
  void write_int(DcmTVR tvr, i64 val);
  void write_flt(DcmTVR tvr, f64 val);
  void write_str(DcmTVR tvr, Str val);
  void write_buf(DcmTVR tvr, Slice<const u8> buf);

  void write_elmt(const DcmElmt& elmt);
  auto read_elmt() -> DcmElmt;

  void write_meta(const DcmMeta& meta);

  template <class T>
  void write_data(const u32 (&shape)[3], Slice<const T> buf);

  template <class T>
  void write_data(const math::NdArray<T, 3>& data) {
    const auto buf = Slice{data.as_ptr(), data.numel()};
    this->write_data(data.shape(), buf);
  }
};

}  // namespace sfc::dcm
