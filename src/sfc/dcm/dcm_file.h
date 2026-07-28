#pragma once

#include "sfc/fs.h"
#include "sfc/dcm/dcm_elmt.h"

namespace sfc::math {
template <class T, u32 N>
struct NdSlice;
}

namespace sfc::dcm {

struct DcmMeta {
  Str SOPClassUID = "1.2.840.10008.5.1.4.1.1.2";  // CT
  Str TransferSyntaxUID = "1.2.840.10008.1.2.1";  // Little Endian
};

struct DcmHead {
  u8 _buff[128] = {};
  char _dicm[4] = {};

  auto is_valid() const noexcept -> bool {
    return Str{_dicm, 4} == "DICM";
  }
};

class DcmFile {
  fs::File _file;

 public:
  DcmFile() noexcept;
  ~DcmFile() noexcept;
  DcmFile(DcmFile&&) noexcept = default;
  DcmFile& operator=(DcmFile&&) noexcept = default;

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
  void write_image(math::NdSlice<T, 2> img);

  template <class T>
  void write_image(math::NdSlice<T, 3> vol);
};

}  // namespace sfc::dcm
