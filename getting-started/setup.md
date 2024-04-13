Connect to remote computer
```
ssh -i ~/.ssh/`key'.pem ubuntu@12345678910
```

Download script
```
wget -O setup_env.sh https://github.com/pharringtonp19/rfp/blob/main/getting-started/setup_env.sh
```

Make script executable
```
chmod +x setup_env.sh
```

Run script
```
./setup_env.sh
```
