apt-get update
apt-get install libboost-all-dev -y
apt search boost -y

apt-get install bison
apt-get install flex -y

pip install -r requirements.txt 

mkdir build
cd build
install_path=`pwd`
cmake .. -DCMAKE_INSTALL_PREFIX=$install_path -DPYTHON_EXECUTABLE=$(which python)
make
make -j install


