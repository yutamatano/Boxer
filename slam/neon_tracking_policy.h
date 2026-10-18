#pragma once
#include <cstddef>

namespace neon {
// ORB-SLAM3 Tracking::eTrackingState: OK=2, RECENTLY_LOST=3.
// A pending reset must still stop the adapter before the NEXT image deletes
// the map. RECENTLY_LOST without a reset can recover via the existing IMU/local
// map tracker. Those frames remain excluded by the trajectory converter.
inline bool shouldFinalizeMetricMap(int state, bool initialized, bool resetPending) {
    return resetPending || !initialized || (state != 2 && state != 3);
}

// If bounded recovery failed, the next fresh session must get the images that
// were only consumed as lost-frame predictions. Match the former immediate
// restart boundary; never replay a successfully recovered/accepted interval.
inline std::size_t nextMetricSessionIndex(std::size_t processedEnd, bool recovering,
                                          std::size_t firstLostIndex) {
    return recovering ? firstLostIndex + 1 : processedEnd;
}
}
