sudo apt update
sudo apt install curl -y
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/master/install.sh | bash
source ~/.bashrc
nvm install --lts

cd next_test_client/med_client
npm i
npm run build
npm run dev
