#! /usr/bin/gnuplot

min_count = 16
point_scale = 0.1

# Fixed Point Size
point_size(n) = n < min_count ? NaN : point_scale

# Proportional Point Size
# point_size(n) = n < min_count ? NaN : ((log(n) - log(min_count) + 1) * point_scale)

centroid_size = 2

splot \
  file index "Colors" using "a":"b":"l":(point_size(column("n"))):"color" title "Colors" with points ps variable pt 5 lc rgbcolor variable, \
  file index "Centroids" using "a":"b":"l":"color" notitle with points ps (centroid_size + 1) pt 7 lc black, \
  file index "Centroids" using "a":"b":"l":"color" title "Centroids" with points ps centroid_size pt 7 lc rgbcolor variable, \

pause mouse close
