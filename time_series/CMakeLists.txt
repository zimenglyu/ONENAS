add_library(exact_time_series time_series.cxx online_series.cxx time_series_episode.cxx)
add_library(online_series online_series.cxx time_series_episode.cxx)

# add_executable(normalize_data normalize_data.cxx)
# target_link_libraries(normalize_data exact_time_series exact_common)

# add_executable(correlation_heatmap correlation_heatmap.cxx)
# target_link_libraries(correlation_heatmap exact_time_series exact_common pthread)
