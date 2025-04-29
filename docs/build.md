## リポジトリとSSH接続設定

自身のプライベートリポジトリで開発する場合の設定
本リポジトリでお試すする場合はスキップする

```
ssh-keygen -o
cat .ssh/id_rsa.pub # Githubに追加する
git clone git@github.com:username/repositoryname.git
```

## Python & Venv

- `pip`と`venv`のインストール
    ```
    sudo apt update
    sudo apt install python3-pip -y
    sudo apt install python3-venv -y
    ```

- 仮想環境の作成＆有効化とパッケージインストール
    ```
    cd diffusion_trial
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

## CUDAドライバインストール

- インストール
    ```
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
    sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
    sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
    sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
    sudo apt-get update
    sudo apt-get -y install cuda
    ```
- パスの登録
    ```
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
    ```