@echo off
echo ========================================================
echo   Instalador Automatico - Motor Hibrido STGCN (CUDA)
echo ========================================================
echo.
echo [ATENCAO] Certifique-se de ter o Python 3.10 instalado.
echo.
pause

echo.
echo [1/5] Criando o Ambiente Virtual (.venv)...
python -m venv .venv

echo.
echo [2/5] Ativando o Ambiente e Atualizando o PIP...
call .venv\Scripts\activate
python -m pip install --upgrade pip

echo.
echo [3/5] Instalando PyTorch Core com Suporte a GPU (CUDA 12.1)...
:: Adicionada flag de timeout estendido para evitar oscilacoes de rede (arquivos de 2.4GB)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --extra-index-url https://download.pytorch.org/whl/cu121 --default-timeout=1000

echo.
echo [4/5] Instalando PyTorch Geometric e Dependencias de Grafos em C++...
pip install torch_cluster==1.6.3 torch_scatter==2.1.2 torch_sparse==0.6.18 torch_spline_conv==1.2.2 -f https://data.pyg.org/whl/torch-2.5.1+cu121.html --default-timeout=1000

echo.
echo [5/5] Instalando o Motor Logico e Interface (requerimentos.txt)...
pip install -r requerimentos.txt

echo.
echo ========================================================
echo Instalacao Concluida! Ambiente pronto para uso.
echo Para trabalhar no projeto, lembre-se de ativar o ambiente:
echo .venv\Scripts\activate
echo ========================================================
pause