from setuptools import setup, find_namespace_packages

def readme():
    with open("README.md") as f:
        return f.read()

setup(
    name='BigMHC',
    version='1.0.1',
    description='BigMHC is a deep learning tool for predicting MHC-I (neo)epitope presentation and immunogenicity.',
    long_description=readme(),
    url='https://github.com/KarchinLab/bigmhc',
    author='Albert, Benjamin Alexander and Yang, Yunxiao and Shao, Xiaoshan M. and Singh, Dipika and Smith, Kellie N. and Anagnostou, Valsamo and Karchin, Rachel',
    scripts=['bigmhc_predict', 'bigmhc_train'],
    packages=find_namespace_packages(),
    include_package_data=True,
    install_requires=[
        'torch',
        'numpy',
        'pandas',
        'psutil',
    ],
    zip_safe=False
)
