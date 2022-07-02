// Martin Kersner, m.kersner@gmail.com
// 2016/12/18

#ifndef NMS_HPP__
#define NMS_HPP__

#include <vector>
#include <numeric>
#include <opencv2/opencv.hpp>

enum PointInRectangle {XMIN, YMIN, XMAX, YMAX};

std::vector<std::pair<std::vector<float>, int>> nms(const std::vector<std::vector<float>> &,
                          const float &);

std::vector<float> GetPointFromRect(const std::vector<std::vector<float>> &,
                                    const PointInRectangle &);

std::vector<float> ComputeArea(const std::vector<float> &,
                               const std::vector<float> &,
                               const std::vector<float> &,
                               const std::vector<float> &);

template <typename T>
std::vector<int> argsort(const std::vector<T> & v);

std::vector<float> Maximum(const float &,
                           const std::vector<float> &);

std::vector<float> Minimum(const float &,
                           const std::vector<float> &);

std::vector<float> CopyByIndexes(const std::vector<float> &,
                                 const std::vector<int> &);

std::vector<int> RemoveLast(const std::vector<int> &);

std::vector<float> Subtract(const std::vector<float> &,
                            const std::vector<float> &);

std::vector<float> Multiply(const std::vector<float> &,
                            const std::vector<float> &);

std::vector<float> Divide(const std::vector<float> &,
                          const std::vector<float> &);

std::vector<int> WhereLarger(const std::vector<float> &,
                             const float &);

std::vector<int> RemoveByIndexes(const std::vector<int> &,
                                 const std::vector<int> &);

std::vector<cv::Rect> BoxesToRectangles(const std::vector<std::vector<float>> &);

template <typename T>
std::vector<std::pair<std::vector<float>, int>> FilterVector(const std::vector<T> &,
                            const std::vector<int> &);

#endif // NMS_HPP__
