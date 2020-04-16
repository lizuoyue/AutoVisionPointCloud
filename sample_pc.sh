PATH="data/2018-10-18-Lim-Chu-Kang-Run-1-Day/point_clouds_length_1000m_overlap_100m/point_cloud_"
for (( i=0; i<=8; ++i ))
do
    zip $PATH"sample_$i.zip" $PATH"$i.txt"
done
# zip *.zip *.txt
# rm *.txt