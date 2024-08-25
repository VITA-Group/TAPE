# 设置需要中断后再次执行的命令
# bash script/restart.sh > restart.log 2>&1 &
name=deepspeed
command='bash script/long_train.sh'
while true; do
    # 使用pgrep查找进程
    if ! pgrep -f $name > /dev/null; then
        # 如果进程不存在，则执行command
        echo "${name} is not running. Executing command..."
        eval $command
    else
        # 如果进程存在，则输出信息
        echo "${name} is currently running. No action taken."
    fi

    # 等待30分钟 (1800秒)
    sleep 1800
done
