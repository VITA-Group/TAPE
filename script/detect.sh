# bash script/detect.sh > detect.log 2>&1 &
detect="zhujiajun"
commands=(
  "bash script/long_train.sh"
)

for cmd in "${commands[@]}"; do
  while true; do
    # 检查进程是否在运行
    if [ $(ps -ef | grep "$detect" | grep -v grep | grep -v code | wc -l)  -gt 0 ];then
      echo "sleep for 5 mins"
      sleep 5m
    else
      echo "run one command: ${cmd}"
      eval $cmd
      sleep 1m
      break; # 退出监控
    fi
  done
done
