测 8080，走 SSH 隧道

服务器端：

cd /home/work/liujingzhi/websocket-test
python3 server_action.py --host 127.0.0.1 --port 8080

客户端机器建立隧道：

ssh -p 34134 -i /data2/liujingzhi/id_ed25519_5090 -N -L 18080:127.0.0.1:8080 root@116.63.180.90

客户端先只检查端口：

python3 /data2/liujingzhi/client_obs_action.py 127.0.0.1 --port 18080 --check-only

正式传输：

python3 /data2/liujingzhi/client_obs_action.py 127.0.0.1 --port 18080 --hz 10