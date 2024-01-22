# downloading and preparing the five datasets
: '
mkdir ../omnidata && cd ../omnidata
wget -c http://cvlab.hanyang.ac.kr/project/omnistereo/data/urban_sunny.zip
unzip urban_sunny.zip && rm urban_sunny.zip

wget -c http://cvlab.hanyang.ac.kr/project/omnistereo/data/urban_cloudy.zip
unzip urban_cloudy.zip && rm urban_cloudy.zip

wget -c http://cvlab.hanyang.ac.kr/project/omnistereo/data/urban_sunset.zip
unzip urban_sunset.zip && rm urban_sunset.zip

wget -c http://cvlab.hanyang.ac.kr/project/omnistereo/data/urban_omnidepth_gt_640.zip
unzip urban_omnidepth_gt_640.zip && rm urban_omnidepth_gt_640.zip
cp -r omnidepth_gt_640 ./sunny/ && cp -r omnidepth_gt_640 ./cloudy/ && mv omnidepth_gt_640 ./sunset/

wget -c http://cvlab.hanyang.ac.kr/project/omnistereo/data/urban_config.yaml
cp urban_config.yaml ./sunny/config.yaml && cp urban_config.yaml ./cloudy/config.yaml && cp urban_config.yaml ./sunset/config.yaml 
rm urban_config.yaml


wget -c http://cvlab.hanyang.ac.kr/project/omnistereo/data/omnihouse.zip
unzip omnihouse.zip && rm omnihouse.zip
wget -c http://cvlab.hanyang.ac.kr/project/omnistereo/data/house_omnidepth_gt_640.zip
unzip house_omnidepth_gt_640.zip && rm house_omnidepth_gt_640.zip
mv omnidepth_gt_640 ./omnihouse/
wget -c http://cvlab.hanyang.ac.kr/project/omnistereo/data/house_config.yaml
mv house_config.yaml ./omnihouse/config.yaml


wget -c http://cvlab.hanyang.ac.kr/project/omnistereo/data/omnithings.zip
unzip omnithings.zip && rm omnithings.zip
wget -c http://cvlab.hanyang.ac.kr/project/omnistereo/data/things_omnidepth_gt_640.zip
unzip things_omnidepth_gt_640.zip && rm things_omnidepth_gt_640.zip
mv omnidepth_gt_640 ./omnithings/
wget -c http://cvlab.hanyang.ac.kr/project/omnistereo/data/things_config.yaml
mv things_config.yaml ./omnithings/config.yaml

mv ./omnithings/cam2/0001.png omnithings/cam2/00001.png
mv ./omnithings/cam2/0002.png omnithings/cam2/00002.png
mv ./omnithings/cam2/0003.png omnithings/cam2/00003.png

mv ./omnithings/cam3/0001.png omnithings/cam3/00001.png
mv ./omnithings/cam3/0002.png omnithings/cam3/00002.png
mv ./omnithings/cam3/0003.png omnithings/cam3/00003.png

mv ./omnithings/cam4/0001.png omnithings/cam4/00001.png
mv ./omnithings/cam4/0002.png omnithings/cam4/00002.png
mv ./omnithings/cam4/0003.png omnithings/cam4/00003.png
'
# The above links are down, as Prof. Jongwoo Lim has moved to SNU from HnagyangU.
# You can download the datasets from https://cuhko365-my.sharepoint.com/:u:/g/personal/217019005_link_cuhk_edu_cn/EUf0Pqx0x31Dj2vXSx4I8WcBsKVslYcyKo4PF_BiJpB-fQ?e=97ycFc, 
# and extract to ../
# Please note that the copyright of the datasets still belongs to the authors of sweepnet#omnimvs, and use these datasets under their licenses.


# getting real samples
git clone https://github.com/hyu-cvlab/omnimvs-pytorch.git
mv omnimvs-pytorch/data/itbt_sample ./
rm -rf omnimvs-pytorch

git clone https://github.com/hyu-cvlab/sweepnet.git
mv sweepnet/real_indoor_sample ./
rm -rf sweepnet






