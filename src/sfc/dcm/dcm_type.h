#pragma once

#include "sfc/core.h"

namespace sfc::dcm {

struct DcmVR {
  char _inn[2] = {};

 public:
  constexpr DcmVR(const char (&s)[3]) noexcept : _inn{s[0], s[1]} {}

  auto is_int() const -> bool;
  auto is_flt() const -> bool;
  auto is_str() const -> bool;
  auto is_buf() const -> bool;

  auto use_len32() const -> bool;

 public:
  constexpr auto operator==(const DcmVR& other) const noexcept -> bool {
    return _inn[0] == other._inn[0] && _inn[1] == other._inn[1];
  }

  void fmt(auto& f) const {
    f.write_str({_inn, 2});
  }
};

struct DcmTag {
  u16 group;
  u16 element;
};

struct DcmTVR {
  DcmTag tag;
  DcmVR vr;

  template <class T>
  static auto PixelData() -> DcmTVR;
};

inline namespace vr {

// numeric / binary VRs
static const DcmVR US = "US";  // u16: unsigned 16-bit
static const DcmVR UL = "UL";  // u32: unsigned 32-bit
static const DcmVR UV = "UV";  // u64: unsigned 64-bit
static const DcmVR SS = "SS";  // i16: signed 16-bit
static const DcmVR SL = "SL";  // i32: signed 32-bit
static const DcmVR SV = "SV";  // i64: signed 64-bit
static const DcmVR FL = "FL";  // f32: IEEE-754 single (32-bit)
static const DcmVR FD = "FD";  // f64: IEEE-754 double (64-bit)
static const DcmVR OB = "OB";  // u8[]: Other Byte
static const DcmVR OW = "OW";  // u16[]: Other Word
static const DcmVR OL = "OL";  // u32[]: Other Long
static const DcmVR OV = "OV";  // u64[]: Other Very Long
static const DcmVR OF = "OF";  // f32[]: Other Float
static const DcmVR OD = "OD";  // f64[]: Other Double
static const DcmVR AT = "AT";  // u16[2]: Attribute Tag (group, element)

// numeric values encoded as strings
static const DcmVR IS = "IS";  // str->i32: Integer String
static const DcmVR DS = "DS";  // str->f64: Decimal String

// character/text VRs
static const DcmVR SH = "SH";  // str[16]: Short String
static const DcmVR LO = "LO";  // str[64]: Long String
static const DcmVR UC = "UC";  // str[]: Unlimited Characters (length varies)

static const DcmVR AS = "AS";  // str[4]: Age String, [0-9]{3}[DWMY]
static const DcmVR AE = "AE";  // str[16]: Application Entity, [-A-Z0-9]*
static const DcmVR CS = "CS";  // str[16]: Code String, [_A-Z]*, eg. PETCT
static const DcmVR PN = "PN";  // str[64]: Person Name
static const DcmVR UI = "UI";  // str[64]: UID, eg. 1.2.826.0.1.3680043.2.1125
static const DcmVR UR = "UR";  // str[]: URI/URL, eg. https://example.com/dicom

// text(multi-line strings)
static const DcmVR ST = "ST";  // str[1024]: Short Text
static const DcmVR LT = "LT";  // str[10240]: Long Text
static const DcmVR UT = "UT";  // str[]: Unlimited Text

// date/time VRs
static const DcmVR TM = "TM";  // Time: HHMMSS(.FFFFFF)
static const DcmVR DA = "DA";  // Date: YYYYMMDD
static const DcmVR DT = "DT";  // DateTime: YYYYMMDDHHMMSS(.FFFFFF)?(+/-ZZZZ)

// others
static const DcmVR SQ = "SQ";  // SQ: Sequence of Items
static const DcmVR UN = "UN";  // UN: Unknown / byte stream
}  // namespace vr

// clang-format off
inline namespace tag {

// Meta
static const DcmTVR GroupLength               = {{0x0002, 0x0000}, UL};  // UL 0002:0000
static const DcmTVR MetaVersion               = {{0x0002, 0x0001}, OB};  // OB 0002:0001
static const DcmTVR MetaSOPClassUID           = {{0x0002, 0x0002}, UI};  // UI 0002:0002
static const DcmTVR TransferSyntaxUID         = {{0x0002, 0x0010}, UI};  // UI 0002:0010

// SOP
static const DcmTVR SOPClassUID               = {{0x0008, 0x0016}, UI};  // UI 0008:0016
static const DcmTVR SOPInstanceUID            = {{0x0008, 0x0018}, UI};  // UI 0008:0018

// Patient [0010]
static const DcmTVR PatientName               = {{0x0010, 0x0010}, PN};  // PN 0010:0010
static const DcmTVR PatientId                 = {{0x0010, 0x0020}, LO};  // LO 0010:0020

// Study[0008, 0020]
static const DcmTVR StudyInstanceUID          = {{0x0020, 0x000D}, UI};   // UI 0020:000D
static const DcmTVR StudyID                   = {{0x0020, 0x0010}, SH};   // SH 0020:0010

// Series
static const DcmTVR Modality                  = {{0x0008, 0x0060}, CS};   // CS 0008:0060 CT|MR|PT
static const DcmTVR SeriesInstanceUID         = {{0x0020, 0x000E}, UI};   // UI 0020:000E
static const DcmTVR SeriesNumber              = {{0x0020, 0x0011}, IS};   // IS 0020:0011
static const DcmTVR ImageNumber               = {{0x0020, 0x0013}, IS};   // IS 0020:0013

// Image [0028]
static const DcmTVR SamplesPerPixel           = {{0x0028, 0x0002}, US};  // US 0028:0002
static const DcmTVR PhotometricInterpretation = {{0x0028, 0x0004}, CS};  // CS 0028:0004
static const DcmTVR NumFrames                 = {{0x0028, 0x0008}, IS};  // IS 0028:0008
static const DcmTVR Rows                      = {{0x0028, 0x0010}, US};  // US 0028:0010
static const DcmTVR Columns                   = {{0x0028, 0x0011}, US};  // US 0028:0011
static const DcmTVR BitsAllocated             = {{0x0028, 0x0100}, US};  // US 0028:0100
static const DcmTVR BitsStored                = {{0x0028, 0x0101}, US};  // US 0028:0101
static const DcmTVR HighBit                   = {{0x0028, 0x0102}, US};  // US 0028:0102
static const DcmTVR PixelRepresentation       = {{0x0028, 0x0103}, US};  // US 0028:0103 uint=0, sint=1
static const DcmTVR RescaleIntercept          = {{0x0028, 0x1052}, DS};  // DS 0028:1052
static const DcmTVR RescaleSlope              = {{0x0028, 0x1053}, DS};  // DS 0028:1053

// Pixel Data [7FE0]
template<class T>
extern const DcmTVR PixelData;

template<>
inline const DcmTVR PixelData<f32>              = {{0x7FE0, 0x0008}, OF};  // OF 7FE0:0008

template<>
inline const DcmTVR PixelData<u8>               = {{0x7FE0, 0x0010}, OB};  // OB 7FE0:0010

template<>
inline const DcmTVR PixelData<u16>              = {{0x7FE0, 0x0010}, OW};  // OW 7FE0:0010
}
// clang-format on

}  // namespace sfc::dcm
