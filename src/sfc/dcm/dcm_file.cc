#include "sfc/fs.h"
#include "sfc/math/ndslice.h"
#include "sfc/dcm/dcm_file.h"

namespace sfc::dcm {

struct TVRL {
  DcmTag tag;
  DcmVR vr;
  u16 len;
};

DcmFile::DcmFile() noexcept {}

DcmFile::~DcmFile() noexcept {}

auto DcmFile::open(Str path) -> io::Result<DcmFile> {
  auto file = _TRY(fs::File::open(path));
  auto head = DcmHead{};
  _TRY(file.read_exact(mem::as_mut_bytes(head)));

  auto res = DcmFile{};
  res._file = mem::move(file);
  return res;
}

auto DcmFile::create(Str path) -> io::Result<DcmFile> {
  auto file = _TRY(fs::File::create(path));
  const auto head = DcmHead{{}, {'D', 'I', 'C', 'M'}};
  _TRY(file.write_all(mem::as_bytes(head)));

  auto res = DcmFile{};
  res._file = mem::move(file);
  return res;
}

void DcmFile::write_int(DcmTVR tvr, i64 val) {
  auto write_imp = [&](auto val) {
    auto buf = Slice{mem::as_bytes(val)};
    this->write_buf(tvr, buf);
  };

  const auto vr = tvr.vr;
  if (vr == dcm::US) return write_imp(u16(val));
  if (vr == dcm::SS) return write_imp(i16(val));
  if (vr == dcm::UL) return write_imp(u32(val));
  if (vr == dcm::SL) return write_imp(i32(val));
  if (vr == dcm::UV) return write_imp(u64(val));
  if (vr == dcm::SV) return write_imp(i64(val));

  auto sval = string::format("{}", val);
  return this->write_str(tvr, sval);
}

void DcmFile::write_flt(DcmTVR tvr, f64 val) {
  auto write_imp = [&](auto val) {
    auto buf = Slice{mem::as_bytes(val)};
    this->write_buf(tvr, buf);
  };

  const auto vr = tvr.vr;
  if (vr == dcm::FL) return write_imp(f32(val));
  if (vr == dcm::FD) return write_imp(f64(val));

  auto sval = string::format("{}", val);
  return this->write_str(tvr, sval);
}

void DcmFile::write_str(DcmTVR tvr, Str val) {
  this->write_buf(tvr, val.as_bytes());
}

void DcmFile::write_buf(DcmTVR tvr, Slice<const u8> buf) {
  const auto [tag, vr] = tvr;

  const auto len64 = num::align_up(buf.len(), u64{2U});
  const auto len32 = num::saturating_cast<u32>(len64);
  const auto len16 = vr.use_len32() ? u16(0U) : u16(len32);
  const auto tvrl = TVRL{tag, vr, len16};
  _file.write_all(mem::as_bytes(tvrl)).unwrap();

  if (vr.use_len32()) {
    _file.write_all(mem::as_bytes(len32)).unwrap();
  }

  _file.write_all(buf).unwrap();
  if (buf.len() % 2 != 0) {
    auto tail = vr.is_str() ? u8{' '} : u8{0};
    _file.write_all(mem::as_bytes(tail)).unwrap();
  }
}

void DcmFile::write_elmt(const DcmElmt& elmt) {
  const auto vr = elmt.vr();
  const auto tag = elmt.tag();
  const auto tvr = DcmTVR{tag, vr};

  if (vr.is_int()) return this->write_int(tvr, elmt.as_int());
  if (vr.is_flt()) return this->write_flt(tvr, elmt.as_flt());
  if (vr.is_str()) return this->write_str(tvr, elmt.as_str());
  return this->write_buf(tvr, elmt.as_buf());
}

auto DcmFile::read_elmt() -> DcmElmt {
  auto tvrl = TVRL{{0, 0}, "UN", 0};
  _file.read_exact(mem::as_mut_bytes(tvrl)).unwrap();
  const auto tvr = DcmTVR{tvrl.tag, tvrl.vr};

  auto len32 = u32{tvrl.len};
  if (tvrl.vr.use_len32()) {
    _file.read_exact(mem::as_mut_bytes(len32)).unwrap();
  }

  const auto vr = tvrl.vr;
  if (vr.is_int()) {
    auto read_int = [&](auto val) {
      _file.read_exact(mem::as_mut_bytes(val)).unwrap();
      return i64(val);
    };

    auto val = i64{0};
    if (vr == dcm::US) val = read_int(u16{});
    if (vr == dcm::SS) val = read_int(i16{});
    if (vr == dcm::UL) val = read_int(u32{});
    if (vr == dcm::SL) val = read_int(i32{});
    if (vr == dcm::UV) val = read_int(u64{});
    if (vr == dcm::SV) val = read_int(i64{});
    return DcmElmt::from_int(tvr, val);
  }

  if (vr.is_flt()) {
    auto read_flt = [&](auto val) {
      _file.read_exact(mem::as_mut_bytes(val)).unwrap();
      return f64(val);
    };

    auto val = f64{0};
    if (vr == dcm::FL) val = read_flt(f32{});
    if (vr == dcm::FD) val = read_flt(f64{});
    return DcmElmt::from_flt(tvr, val);
  }

  const auto buf_cap = num::align_up(len32, 2U);
  if (vr.is_str()) {
    auto val = String::with_capacity(buf_cap);

    auto& buf = val.as_mut_buf();
    _file.read_exact(buf.spare_capacity_mut()).unwrap();
    buf.set_len(len32);
    return DcmElmt::from_str(tvr, mem::move(val));
  }

  auto val = List<u8>::with_capacity(buf_cap);
  _file.read_exact(val.spare_capacity_mut()).unwrap();
  return DcmElmt::from_buf(tvr, mem::move(val));
}

void DcmFile::write_meta(const DcmMeta& meta) {
  auto meta_ver = List<u8>{};
  meta_ver.push(0);
  meta_ver.push(1);

  const DcmElmt elmts[] = {
      DcmElmt::from_buf(tag::MetaVersion, mem::move(meta_ver)),
      DcmElmt::from_str(tag::MetaSOPClassUID, String::from(meta.SOPClassUID)),
      DcmElmt::from_str(tag::TransferSyntaxUID, String::from(meta.TransferSyntaxUID)),
  };

  auto group_len = 0U;
  for (const auto& elmt : elmts) {
    group_len += elmt.serialize_size();
  }

  const auto group = DcmElmt::from_int(tag::GroupLength, group_len);
  this->write_elmt(group);
  for (const auto& elmt : elmts) {
    this->write_elmt(elmt);
  }
}

template <class T>
void DcmFile::write_image(math::NdSlice<T, 2> img) {
  sfc::assert_(img.is_contiguous(), "DcmFile::write_img: img must be contiguous");

  const auto nframe = 1U;
  const auto [nrow, ncol] = img._shape;
  const auto pixel_representation = sfc::trait::sint_<T> ? 1 : 0;

  this->write_int(tag::SamplesPerPixel, 1);                        // gray
  this->write_str(tag::PhotometricInterpretation, "MONOCHROME2");  // gray
  this->write_int(tag::NumFrames, nframe);
  this->write_int(tag::Rows, nrow);
  this->write_int(tag::Columns, ncol);
  this->write_int(tag::BitsAllocated, 8 * sizeof(T));
  this->write_int(tag::BitsStored, 8 * sizeof(T));
  this->write_int(tag::HighBit, 8 * sizeof(T) - 1);
  this->write_int(tag::PixelRepresentation, pixel_representation);

  const auto pixel_data_tvr = DcmTVR::PixelData<T>();

  auto buf = Slice{img._data, img.numel()};
  this->write_buf(pixel_data_tvr, buf.as_bytes());
}

template <class T>
void DcmFile::write_image(math::NdSlice<T, 3> vol) {
  sfc::assert_(vol.is_contiguous(), "DcmFile::write_img: img must be contiguous");

  const auto [nframe, nrow, ncol] = vol._shape;
  const auto pixel_representation = sfc::trait::sint_<T> ? 1 : 0;

  this->write_int(tag::SamplesPerPixel, 1);                        // gray
  this->write_str(tag::PhotometricInterpretation, "MONOCHROME2");  // gray
  this->write_int(tag::NumFrames, nframe);
  this->write_int(tag::Rows, nrow);
  this->write_int(tag::Columns, ncol);
  this->write_int(tag::BitsAllocated, 8 * sizeof(T));
  this->write_int(tag::BitsStored, 8 * sizeof(T));
  this->write_int(tag::HighBit, 8 * sizeof(T) - 1);
  this->write_int(tag::PixelRepresentation, pixel_representation);

  const auto pixel_data_tvr = DcmTVR::PixelData<T>();

  auto buf = Slice{vol._data, vol.numel()};
  this->write_buf(pixel_data_tvr, buf.as_bytes());
}

template void DcmFile::write_image(math::NdSlice<u8, 2>);
template void DcmFile::write_image(math::NdSlice<u16, 2>);
template void DcmFile::write_image(math::NdSlice<f32, 2>);

template void DcmFile::write_image(math::NdSlice<u8, 3>);
template void DcmFile::write_image(math::NdSlice<u16, 3>);
template void DcmFile::write_image(math::NdSlice<f32, 3>);

}  // namespace sfc::dcm
