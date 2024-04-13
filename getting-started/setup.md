Connect to remote 
```
ssh -i ~/.ssh/`key'.pem ubuntu@12345678910
```

```
wget -O setup_env.sh https://raw.githubusercontent.com/pharringtonp19/rfp/main/getting-started/setup_env.sh
```

Clone Repository and Open Getting-Started Folder
```
git clone https://github.com/pharringtonp19/rfp.git && cd rfp/getting-started
```

Make script executable
```
chmod +x setup_env.sh
```

Run script
```
./setup_env.sh
```
