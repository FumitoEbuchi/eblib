# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py
import os, sys
from setuptools import setup, find_packages
import site
import glob
import os.path

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

def read_requirements():
    """Parse requirements from requirements.txt."""
    reqs_path = os.path.join('.', 'requirements.txt')
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements
# site-packageディレクトリのパスを取得
# ※リストの先頭に"C:\Python34"が入ってるみたいなので最後がsite-packageだと想定して処理します（確実ではなさそうなのでいい方法があったら教えてください）
sitedir = site.getsitepackages()[-1]
print(sitedir)
# インストール先のディレクトリ
installdir = os.path.join(sitedir, 'eblib')

# インストール元のモジュールのルートディレクトリを取得
# ※setup.pyからの相対パスで取得します
mydir = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'eblib')
# データファイルのリストを取得
real_bridge_wav = glob.glob(os.path.join(mydir, 'dataset', '*.wav'))
real_bridge_csv = glob.glob(os.path.join(mydir, 'dataset', '*.csv'))
test_bridge_wav = glob.glob(os.path.join(mydir, 'dataset/test_pieces', '*.WAV'))
test_bridge_csv = glob.glob(os.path.join(mydir, 'dataset/test_pieces', '*.csv'))
setup(
    name='eblib',
    version='0.0.1',
    description='Machine Learning package for Anomaly Detection',
    long_description=readme,
    author='Fumito Ebuchi',
    author_email='fumito.ebuchi@gmail.com',
    url='',
    license=license,
    install_requires=read_requirements(),
    packages=find_packages(exclude=('tests', 'docs')),
    include_package_data=True,
    data_files=[(os.path.join(installdir, 'dataset'), real_bridge_wav),
    (os.path.join(installdir, 'dataset'), real_bridge_csv),
    (os.path.join(installdir, 'dataset/test_pieces'), test_bridge_wav),
    (os.path.join(installdir, 'dataset/test_pieces'), test_bridge_csv)]
)

