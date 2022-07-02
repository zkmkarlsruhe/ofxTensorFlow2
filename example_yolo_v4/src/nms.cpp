// Martin Kersner, m.kersner@gmail.com
// 2016/12/18
// C++ version of http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

#include "nms.hpp"
using std::vector;
using cv::Rect;
using cv::Point;

vector<std::pair<std::vector<float>,int>> nms(const vector<vector<float>> & boxes,
                 const float & threshold)
{
  if (boxes.empty())
  	return vector<std::pair<std::vector<float>, int>>();
  
  // grab the coordinates of the bounding boxes
  auto x1 = GetPointFromRect(boxes, XMIN);
  auto y1 = GetPointFromRect(boxes, YMIN);
  auto x2 = GetPointFromRect(boxes, XMAX);
  auto y2 = GetPointFromRect(boxes, YMAX);
  
  // compute the area of the bounding boxes and sort the bounding
  // boxes by the bottom-right y-coordinate of the bounding box
  auto area = ComputeArea(x1, y1, x2, y2);
  auto idxs = argsort(y2);
  
  int last;
  int i;
  vector<int> pick;
  
  // keep looping while some indexes still remain in the indexes list
  while (idxs.size() > 0) {
    // grab the last index in the indexes list and add the
    // index value to the list of picked indexes
    last = idxs.size() - 1;	
    i    = idxs[last];
    pick.push_back(i);
    
    // find the largest (x, y) coordinates for the start of
    // the bounding box and the smallest (x, y) coordinates
    // for the end of the bounding box
    auto idxsWoLast = RemoveLast(idxs);

    auto xx1 = Maximum(x1[i], CopyByIndexes(x1, idxsWoLast));
    auto yy1 = Maximum(y1[i], CopyByIndexes(y1, idxsWoLast));
    auto xx2 = Minimum(x2[i], CopyByIndexes(x2, idxsWoLast));
    auto yy2 = Minimum(y2[i], CopyByIndexes(y2, idxsWoLast));

		// compute the width and height of the bounding box
    auto w = Maximum(0, Subtract(xx2, xx1));
    auto h = Maximum(0, Subtract(yy2, yy1));
		
		// compute the ratio of overlap
    auto overlap = Divide(Multiply(w, h), CopyByIndexes(area, idxsWoLast));

    // delete all indexes from the index list that have
    auto deleteIdxs = WhereLarger(overlap, threshold);
    deleteIdxs.push_back(last);
    idxs = RemoveByIndexes(idxs, deleteIdxs);
  }

  return FilterVector(boxes, pick);
}

vector<float> GetPointFromRect(const vector<vector<float>> & rect,
                               const PointInRectangle & pos)
{
  vector<float> points;
  
  for (const auto & p: rect)
    points.push_back(p[pos]);
  
  return points;
}

vector<float> ComputeArea(const vector<float> & x1,
                          const vector<float> & y1,
                          const vector<float> & x2,
                          const vector<float> & y2)
{
  vector<float> area;
  auto len = x1.size();
  
  for (decltype(len) idx = 0; idx < len; ++idx) {
    auto tmpArea = (x2[idx] - x1[idx] + 1) * (y2[idx] - y1[idx] + 1);
    area.push_back(tmpArea);
  }
  
  return area;
}

template <typename T>
vector<int> argsort(const vector<T> & v)
{
  // initialize original index locations
  vector<int> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);
  
  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(),
       [&v](int i1, int i2) {return v[i1] < v[i2];});
  
  return idx;
}

vector<float> Maximum(const float & num,
                      const vector<float> & vec)
{
  auto maxVec = vec;
  auto len = vec.size();
  
  for (decltype(len) idx = 0; idx < len; ++idx)
    if (vec[idx] < num)
      maxVec[idx] = num;
  
  return maxVec;
}

vector<float> Minimum(const float & num,
                      const vector<float> & vec)
{
  auto minVec = vec;
  auto len = vec.size();
  
  for (decltype(len) idx = 0; idx < len; ++idx)
    if (vec[idx] > num)
      minVec[idx] = num;
  
  return minVec;
}

vector<float> CopyByIndexes(const vector<float> & vec,
                            const vector<int> & idxs)
{
  vector<float> resultVec;
  
  for (const auto & idx : idxs)
    resultVec.push_back(vec[idx]);
  
  return resultVec;
}

vector<int> RemoveLast(const vector<int> & vec)
{
  auto resultVec = vec;
  resultVec.erase(resultVec.end()-1);
  return resultVec;
}

vector<float> Subtract(const vector<float> & vec1,
                       const vector<float> & vec2)
{
  vector<float> result;
  auto len = vec1.size();
  
  for (decltype(len) idx = 0; idx < len; ++idx)
    result.push_back(vec1[idx] - vec2[idx] + 1);
  
  return result;
}

vector<float> Multiply(const vector<float> & vec1,
		                   const vector<float> & vec2)
{
  vector<float> resultVec;
  auto len = vec1.size();
  
  for (decltype(len) idx = 0; idx < len; ++idx)
    resultVec.push_back(vec1[idx] * vec2[idx]);
  
  return resultVec;
}

vector<float> Divide(const vector<float> & vec1,
		                 const vector<float> & vec2)
{
  vector<float> resultVec;
  auto len = vec1.size();
  
  for (decltype(len) idx = 0; idx < len; ++idx)
    resultVec.push_back(vec1[idx] / vec2[idx]);
  
  return resultVec;
}

vector<int> WhereLarger(const vector<float> & vec,
                        const float & threshold)
{
  vector<int> resultVec;
  auto len = vec.size();
  
  for (decltype(len) idx = 0; idx < len; ++idx)
    if (vec[idx] > threshold)
      resultVec.push_back(idx);
  
  return resultVec;
}

vector<int> RemoveByIndexes(const vector<int> & vec,
                            const vector<int> & idxs)
{
  auto resultVec = vec;
  auto offset = 0;
  
  for (const auto & idx : idxs) {
    resultVec.erase(resultVec.begin() + idx + offset);
    offset -= 1;
  }
  
  return resultVec;
}

vector<Rect> BoxesToRectangles(const vector<vector<float>> & boxes)
{
  vector<Rect> rectangles;
  vector<float> box;
  
  for (const auto & box: boxes)
    rectangles.push_back(Rect(Point(box[0], box[1]), Point(box[2], box[3])));
  
  return rectangles;
}

template <typename T>
std::vector<std::pair<std::vector<float>,int>> FilterVector(const vector<T> & vec,
    const vector<int> & idxs)
{
    vector<std::pair<std::vector<float>, int>> resultVec;
    for (const auto& idx : idxs) {
        resultVec.push_back(std::make_pair(vec[idx], idx));
    }
  return resultVec;
}
