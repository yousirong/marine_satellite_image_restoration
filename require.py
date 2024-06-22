# requirement.txt 버전 안맞는 패키지 제외하고 가능한 패키지만 다운로드
import subprocess
import sys

def install_packages(requirements_file):
    with open(requirements_file, 'r') as file:
        for line in file:
            package = line.strip()
            if package and not package.startswith('#'):
                try:
                    print(f"Installing {package}...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                except subprocess.CalledProcessError:
                    print(f"Failed to install {package}. Skipping...")

install_packages('requirements.txt')
