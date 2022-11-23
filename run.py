from typing import Any, List, Optional, Tuple
import numpy as np
from numpy import typing as np_type
import os
from PIL import Image
from PIL.Image import Image as ImageType


class Imagem:
    def __init__(self) -> None:
        self.data = None
        self.type = None
        self.shape = None

    def _abrir_imagem(self, caminho_arquivo:str) -> bool:
        try:
            imagem:ImageType = Image.open(caminho_arquivo)
            self.data = np.array(imagem.convert('L'))
            self.type = type(self.data)
            self.shape = self.data.shape
            return True
        except Exception:
            return False

    def converter_np_array(self, caminho_arquivo:str):
        total = np.array([])
        if self._abrir_imagem(caminho_arquivo):
            for linha in self.data:
                total = np.append(total, linha)
        return total


class Parametro:
    def __init__(
        self,
        imagem: Imagem,
        apredizagem: float= 0.1,
        epocas: int= 1,
        momento: int= 1,
        sub_pasta: str='',
        qtd_neuronios_camada_oculta: int= 1,
        imagem_unica: str = ''
    ) -> None:
        self.arquivos = []
        self.entradas = []
        self.saidas = []
        self.imagem = imagem
        self.apredizagem = apredizagem
        self.epocas = epocas
        self.momento = momento
        self.qtd_neuronios_camada_oculta = qtd_neuronios_camada_oculta
        self.caminho_entradas = f'assets/{sub_pasta}'
        self.imagem_unica = imagem_unica


    def _listar_arquivos(self) -> None:
        for _, _, files in os.walk(os.path.abspath(self.caminho_entradas)):
            self.arquivos.extend(iter(files))


    def _ler_imagens(self) -> None:
        tamanho_nome_img = None
        for nome_imagem_com_extensao in self.arquivos:
            if tamanho_nome_img is None:
                tamanho_nome_img = len(nome_imagem_com_extensao)
            elif len(nome_imagem_com_extensao) != tamanho_nome_img:
                raise ValueError(
                    f'Nome da imagem com tamanho diferente! {nome_imagem_com_extensao}'
                )
            self._montar_entradas(nome_imagem_com_extensao)
            self._montar_saidas(nome_imagem_com_extensao)


    def _montar_entradas(self, nome_imagem_com_extensao:str) -> None:
        entrada = self.imagem.converter_np_array(
            caminho_arquivo=f'{self.caminho_entradas}{nome_imagem_com_extensao}'
        )
        self.entradas.append(entrada)


    def _montar_saidas(self, nome_imagem_com_extensao:str) -> None:
        nome_imagem = nome_imagem_com_extensao.replace('.png', '')

        # imagem com 1 caracter
        if '_' in nome_imagem:
            caracter_entrada, _ = nome_imagem.split('_')
            binario = self._caracter_para_binario(caracter_entrada)
            saida_numero = self._binario_para_saida_esperada_ativacao(binario)

        # imagem com mais de 1 caracter
        else:
            saida_numero = np.array([])
            for caracter in nome_imagem:
                binario = self._caracter_para_binario(caracter)
                saida_atual = self._binario_para_saida_esperada_ativacao(binario)
                saida_numero = np.concatenate((saida_numero, np.array(saida_atual)))

        self.saidas.append(saida_numero)


    def _binario_para_saida_esperada_ativacao(self, binario:str):
        return [-1 if int(bit) == 0 else 1 for bit in binario]

    def _caracter_para_binario(self, caracter:str) -> str:
        try:
            convertido = int(caracter)
        except ValueError:
            convertido = ord(caracter)
        return format(convertido, '08b')

    def executar(self) -> Tuple[
        List[np_type.NDArray[Any]],
        List[List[Any]]
    ]:

        self._listar_arquivos()
        self._ler_imagens()
        return self.entradas, self.saidas

class TreinamentoPerceptronMultiCamadas:
    def __init__(self, parametro: Parametro) -> None:
        self._parametro = parametro
        self.apredizagem = self._parametro.apredizagem
        self.epocas = self._parametro.epocas
        self.qtd_neuronios_camada_oculta = self._parametro.qtd_neuronios_camada_oculta
        self.momento = self._parametro.momento

        self.resultados  = None
        self.medias_absolutas = []


    # função de ativação tanh
    def _ativar(self, valor: Any) -> Any:
        return np.tanh(valor)

    def _derivar(self, valor: Any) -> Any:
        return 1.0 - np.tanh(valor)**2

    def _delta_saida(self, erro_camada_saida, camada_saida_ativada):
        return np.multiply(
            erro_camada_saida,
            self._derivar(camada_saida_ativada)
        )

    def _delta_oculta(self, delta_saida, camada_oculta_ativada):
        return np.multiply(
            delta_saida.dot(self.pesos_camada_saida.T),
            self._derivar(camada_oculta_ativada)
        )

    def _novos_pesos(self, pesos_antigos, pesos_novos):
        return np.add(
            np.multiply(pesos_antigos, self.momento),
            np.multiply(pesos_novos, self.apredizagem)
        )


    def _gerar_pesos(self) -> None:
        self._parametro.executar()
        self.entradas = np.array(self._parametro.entradas)
        self.saidas = np.array(self._parametro.saidas)

        self.pesos_camada_oculta:np_type.NDArray = 2 * np.random.random((
            len(self.entradas[0]), # qtd linhas
            self.qtd_neuronios_camada_oculta, # qtd colunas
        )) -1 # torna negativo

        self.pesos_camada_saida:np_type.NDArray = 2 * np.random.random((
            self.qtd_neuronios_camada_oculta, # qtd linhas
            len(self.saidas[0]), # qtd colunas
        )) -1 # torna negativo


    def executar(self) -> float:
        self._gerar_pesos()

        porcentagem_erro = 0.0
        for index in range(self.epocas):
            soma_sinapse_oculta   = np.dot(self.entradas, self.pesos_camada_oculta)
            camada_oculta_ativada = self._ativar(soma_sinapse_oculta)
            soma_sinapse_saida    = np.dot(camada_oculta_ativada, self.pesos_camada_saida)
            camada_saida_ativada  = self._ativar(soma_sinapse_saida)
            self.resultados       = camada_saida_ativada

            # calculo do erro
            erro_camada_saida = np.subtract(self.saidas, camada_saida_ativada)
            media_absoluta    = np.mean(np.abs(erro_camada_saida))
            self.medias_absolutas.append((media_absoluta*100))
            porcentagem_erro = float(media_absoluta*100)

            print('porcentagem_erro', format(porcentagem_erro, 'f'))

            # calcula Deltas
            delta_saida  = self._delta_saida(erro_camada_saida, camada_saida_ativada)
            delta_oculta = self._delta_oculta(delta_saida, camada_oculta_ativada)

            # calcula novos pesos da camada de saida
            pesos_saida_novos = camada_oculta_ativada.T.dot(delta_saida)
            self.pesos_camada_saida = self._novos_pesos(self.pesos_camada_saida, pesos_saida_novos)

            # calcula novos pesos da camada oculta
            pesos_oculta_novos       = self.entradas.T.dot(delta_oculta)
            self.pesos_camada_oculta = self._novos_pesos(self.pesos_camada_oculta, pesos_oculta_novos)

        return porcentagem_erro

for _ in range(1, 10):
    treinamento = TreinamentoPerceptronMultiCamadas(
        parametro=Parametro(
            imagem=Imagem(),
            apredizagem=0.2,
            epocas=500,
            qtd_neuronios_camada_oculta=150,
            sub_pasta='placas/',
            momento=1,
        ),
    )
    porc_erro_atual = treinamento.executar()
