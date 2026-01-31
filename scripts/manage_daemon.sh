#!/bin/bash
# 守护进程管理脚本

case "$1" in
    start)
        echo "启动自动同步守护进程..."
        nohup bash /workspace/projects/scripts/auto_sync_daemon.sh > /workspace/backups/daemon.log 2>&1 &
        sleep 1
        if ps aux | grep auto_sync_daemon | grep -v grep > /dev/null; then
            echo "✓ 守护进程已启动"
        else
            echo "✗ 守护进程启动失败"
        fi
        ;;
    stop)
        echo "停止自动同步守护进程..."
        pkill -f auto_sync_daemon
        sleep 1
        if ps aux | grep auto_sync_daemon | grep -v grep > /dev/null; then
            echo "✗ 守护进程停止失败"
        else
            echo "✓ 守护进程已停止"
        fi
        ;;
    restart)
        echo "重启自动同步守护进程..."
        $0 stop
        sleep 1
        $0 start
        ;;
    status)
        if ps aux | grep auto_sync_daemon | grep -v grep > /dev/null; then
            echo "✓ 守护进程正在运行"
            echo "PID: $(ps aux | grep auto_sync_daemon | grep -v grep | awk '{print $2}')"
            echo ""
            echo "最近的日志:"
            tail -10 /workspace/backups/daemon.log
        else
            echo "✗ 守护进程未运行"
        fi
        ;;
    log)
        echo "守护进程日志:"
        tail -30 /workspace/backups/daemon.log
        ;;
    *)
        echo "用法: $0 {start|stop|restart|status|log}"
        exit 1
        ;;
esac
