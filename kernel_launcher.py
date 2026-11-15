"""Jupyter 커널 실행기: Isaac Sim 내장 파이썬(python.sh) + cuRobo + 별도 설치한 ipykernel.

~/.local/share/jupyter/kernels/isaacsim-curobo/kernel.json 이 이 파일을 python.sh로 실행한다.
.kernel_pkgs(ipykernel 등)는 sys.path 맨 뒤에 붙여서, Isaac Sim에 이미 있는 패키지
(tornado, typing_extensions, packaging ...)는 Isaac Sim 버전이 그대로 쓰이게 한다.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKGS = str(HERE / '.kernel_pkgs')
sys.path.insert(0, str(Path.home() / 'curobo' / 'src'))

# jupyter_client는 Isaac Sim 번들보다 새 typing_extensions가 필요하다(extra_items).
# 이것만 먼저 새 버전으로 불러 두고, 나머지는 Isaac Sim 버전이 우선하도록 맨 뒤에 붙인다.
sys.path.insert(0, PKGS)
import typing_extensions  # noqa: E402,F401
sys.path.remove(PKGS)
sys.path.append(PKGS)

from ipykernel import kernelapp

kernelapp.launch_new_instance()
