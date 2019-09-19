docker build -t registry-vpc.cn-shanghai.aliyuncs.com/shuzhi/matting-docker-gpu:$1 -f docker/docker_image_gpu/Dockerfile .
docker build -t registry-vpc.cn-shanghai.aliyuncs.com/shuzhi/matting-docker:$1 -f docker/docker_image/Dockerfile .

docker push registry-vpc.cn-shanghai.aliyuncs.com/shuzhi/matting-docker:$1
docker push registry-vpc.cn-shanghai.aliyuncs.com/shuzhi/matting-docker-gpu:$1