#include "gemv.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

nb::ndarray<int32_t, nb::numpy, nb::c_contig>
compute_gemv(nb::ndarray<uint8_t, nb::numpy, nb::c_contig> weight,
             nb::ndarray<int8_t, nb::numpy, nb::c_contig> activation,
             int n_rows, int n_cols) {
  // 1. Cấp phát bộ nhớ thô trong C++ cho kết quả đầu ra
  int32_t *data = new int32_t[n_rows];

  // 2. Thực thi hàm C++ AVX2 (ghi thẳng kết quả vào mảng data vừa tạo)
  gemv_i2_s((const uint8_t *)weight.data(), (const int8_t *)activation.data(),
            data, n_rows, n_cols);

  // 3. Khai báo số chiều của mảng (Shape)
  size_t shape[1] = {static_cast<size_t>(n_rows)};

  // 4. Tạo Capsule: Báo cho Python biết cách giải phóng bộ nhớ khi mảng bị dọn
  // rác (Garbage Collector)
  nb::capsule owner(data, [](void *p) noexcept { delete[] (int32_t *)p; });

  // 5. Khởi tạo và trả về Numpy array không copy (zero-copy)
  return nb::ndarray<int32_t, nb::numpy, nb::c_contig>(
      data,  // con trỏ dữ liệu
      1,     // số chiều (ndim)
      shape, // mảng mô tả kích thước
      owner  // cơ chế quản lý vòng đời
  );
}

NB_MODULE(_gemv, m) { m.def("compute_gemv", &compute_gemv); }