##  Resultados de qualidade


| Métrica | Heurístico Alterado | Heurístico Original |
|---|---:|---:|
| GT | 8.000 | 8.000 |
| Predições | 7.408 | 1.360 |
| TP @ IoU 0,5 | 5.120 | 1.016 |
| FP | 2.288 | 344 |
| FN | 2.880 | 6.984 |
| Precisão | 0,6911 | 0,7471 |
| Recall | 0,6400 | 0,1270 |
| F1 derivado | 0,6646 | 0,2171 |
| Mean best IoU | 0,476 | 0,115 |
| OCR exato | 4.239 | 851 |
| OCR/all | 0,5299 | 0,1064 |
| OCR/matched | 0,8279 | 0,8376 |
| Chamadas de OCR por imagem | 5,556 | 1,020 |

##  Latência e throughput

| Métrica | Heurístico Alterado  | Heurístico Original |
|---|---:|---:|
| Detecção de veículo média | 9,747 ms | 8,233 ms |
| Detecção de placa média | 11,817 ms | 6,225 ms |
| Detecção de placa P95 | 27,2 ms | 12,7 ms |
| Detecção de placa P99 | 40,1 ms | 18,1 ms |
| OCR médio | 13,592 ms | 2,302 ms |
| ALPR total médio | 25,594 ms | 8,582 ms |
| Pipeline total médio | 35,341 ms | 16,814 ms |
| Throughput global | 18,0 FPS | 27,8 FPS |
| Duração real | 443,922 s | 287,547 s |

##  Diferenças

========-
| Aspecto | Heurístico Alterado  | Heurístico Original |
|---|---|---|
| Escala de entrada | Normaliza o crop para largura 512 | Mantém a resolução recebida |
| Gradiente | Scharr X | Sobel X e Y, com magnitude |
| Contraste de caracteres | Blackhat e máscara de regiões claras | CLAHE e Otsu sobre gradientes |
| Morfologia | Kernels direcionais por família | Abertura e fechamento `3x3` genéricos |
| Área | Limites relativos globais e por família | 0,1% a 4% para ambas as famílias |
| Proporção horizontal | 2,35 a 3,65 | 2,2 a 4,0 |
| Proporção de motocicleta | 0,85 a 1,60 | 1,0 a 1,5 |
| Candidatos detalhados | Até 24 após deduplicação | Cinco por busca |
| Evidências de caracteres | Quantidade, altura, alinhamento e cobertura | Quantidade, proporção e área absoluta |
| Ranking | Geometria, área, caracteres, posição e contraste | Densidade, proporção, posição baixa e centralidade |
| Threshold adaptativo | Fallback quando Otsu é insuficiente | Não utilizado |
| Ordem das famílias | Avaliadas no mesmo fluxo | Carro primeiro; moto somente como fallback |
| Saída por crop | Melhor candidato acima de 0,60 | Primeiro candidato com pelo menos 5 caracteres |


##  Decisões de projeto do heurístico Alterado

O detector foi desenhado para aumentar a cobertura sem recorrer a um modelo
treinado. A decisão central foi substituir uma sequência de filtros rígidos por
um processo em duas etapas: primeiro gerar candidatos com restrições
geométricas amplas; depois ordená-los combinando evidências independentes. Isso
permite que uma placa com iluminação, escala ou posição imperfeita continue no
pipeline quando as demais características forem fortes.

###  Processar crops de veículos em vez do frame completo

O heurístico recebe cada região detectada pelo YOLOX-S separadamente. Essa
decisão reduz o espaço de busca, aumenta a área relativa ocupada pela placa e
torna mais razoáveis as hipóteses de posição e escala usadas no score. Ao final,
as coordenadas da placa são traduzidas de volta para o frame original.

O trade-off é uma dependência forte do detector veicular: se o YOLOX não gerar
um crop correto, o heurístico não terá oportunidade de encontrar a placa. Por
isso, a qualidade ponta a ponta não pode ser atribuída somente ao código de
placas.

###  Normalizar a largura para 512 pixels

Os crops recebidos variam de resolução conforme distância, classe e tamanho do
veículo. O redimensionamento para largura 512, preservando a razão de aspecto,
faz os kernels morfológicos e os tamanhos mínimos operarem em uma escala mais
estável. `INTER_CUBIC` é usado ao ampliar e `INTER_AREA` ao reduzir, escolhas
adequadas respectivamente para interpolação e redução de aliasing.

A largura 512 é um compromisso entre retenção de detalhes de caracteres e
custo de CPU. O valor não é aprendido: aumentar a largura tende a preservar
placas pequenas, mas eleva o número de pixels processados; reduzi-la economiza
tempo, mas pode apagar traços finos.

### Aplicar Median Blur antes do CLAHE

O filtro de mediana `3x3` reduz ruído impulsivo sem suavizar bordas tanto quanto
um filtro linear. Em seguida, o CLAHE com `clipLimit=3,0` realça contraste
local, necessário quando placa e carro possuem iluminação desigual.

A ordem foi escolhida para evitar que o CLAHE amplifique primeiro o ruído. O
limite 3,0 é mais conservador que o 5,0 usado no Original: contraste excessivo pode
criar bordas espúrias que posteriormente se transformam em falsos candidatos.

### Usar Black Hat e Scharr apenas no eixo X

Caracteres de placas normalmente aparecem como estruturas escuras sobre uma
região mais clara. A operação Black Hat destaca exatamente essa diferença entre
o fechamento morfológico e a imagem. Depois, o Scharr em X responde às bordas
verticais repetidas dos caracteres.

Essa combinação é mais específica que usar a magnitude de Sobel X e Y: bordas
horizontais da carroceria, para-choques e sombras recebem menos importância. O
Scharr também oferece uma aproximação de derivada mais isotrópica para kernel
pequeno. A especialização melhora a formação de grupos de caracteres, mas pode
perder placas com contraste invertido, reflexo forte ou orientação muito
inclinada.

### Intersectar com uma máscara de regiões claras

Uma máscara produzida por fechamento e Otsu restringe a resposta direcional a
regiões predominantemente claras. A intenção é rejeitar texturas do veículo
que também possuam muitas bordas verticais, mas não se pareçam com o fundo de
uma placa.

Essa decisão reduz falsos candidatos em carroceria, grade e cenário. Em
contrapartida, introduz uma hipótese de aparência: placas muito escuras,
subexpostas ou parcialmente sombreadas podem ser eliminadas cedo.

### Usar kernels direcionais e múltiplas expansões

Os kernels retangulares `13x5` e `21x5`, definidos na referência de largura
640 e escalados para a largura normalizada, fecham intervalos entre traços
vizinhos. Duas larguras permitem responder a espaçamentos diferentes sem
depender de um único tamanho estrutural.

Cada contorno gera caixas com mais de uma expansão. A máscara tende a cobrir os
caracteres, não necessariamente toda a borda física da placa; expandir em X e Y
busca recuperar margens cortadas e melhorar o IoU com a anotação. O custo é
gerar hipóteses sobrepostas, posteriormente tratadas pela deduplicação.

### Separar placas horizontais e de motocicleta

Uma única faixa de proporção não representa bem placas automotivas horizontais
e placas de motocicleta. Por isso o detector mantém duas famílias:

- horizontal: razão de 2,35 a 3,65, alvo 2,90;
- motocicleta: razão de 0,85 a 1,60, alvo 1,15.

Cada família também possui área-alvo, expansões e variantes morfológicas
próprias. Para motocicletas há uma variante com fechamento `31x35`, destinada
a agrupar uma disposição mais alta e próxima do quadrado. As duas famílias são
avaliadas no mesmo fluxo para evitar que um falso candidato horizontal impeça
a tentativa da geometria de moto.

Os intervalos mais largos favorecem recall. Eles também admitem mais objetos
não placa, razão pela qual geometria isolada não decide o resultado final.

### Usar área relativa e erro logarítmico

Área em pixels seria dependente da resolução. O detector usa a fração da imagem
ocupada pelo candidato e combina limites globais com limites específicos de
cada família. Isso torna a regra comparável entre crops de tamanhos distintos.

Na pontuação, a distância para a área-alvo é calculada no domínio logarítmico.
Assim, uma área duas vezes maior e uma área duas vezes menor recebem erros de
magnitude semelhante. A área continua sendo uma hipótese sobre a composição do
crop, mas deixa de ser uma barreira absoluta ligada ao número de pixels.

### Fazer pré-ranking antes da análise detalhada

Todos os contornos aprovados recebem primeiro um score barato:

```text
0,40 × razão + 0,45 × área + 0,15 × preenchimento do contorno
```

Somente os melhores candidatos seguem para a análise estrutural de caracteres,
limitados a 24 após deduplicação. A decisão controla o custo do
`connectedComponentsWithStats` e do threshold adaptativo sem escolher o
vencedor apenas pela geometria.

O limite 24 prioriza latência previsível. Em cenas muito poluídas, porém, um
candidato correto mal ranqueado pode ser descartado antes da pontuação final.

### Deduplicar candidatos com IoU de 0,90

As expansões e kernels diferentes podem produzir várias caixas quase idênticas.
O detector remove uma caixa quando ela pertence à mesma família e tem
`IoU >= 0,90` com outra já preservada.

O limiar alto evita eliminar alternativas que ainda diferem materialmente no
enquadramento. A deduplicação é feita dentro de cada família; portanto,
hipóteses horizontal e motocicleta podem coexistir sobre a mesma região e
competir no score final.

### Integrar evidências de caracteres ao ranking

Em vez de escolher uma caixa e só depois verificar se ela contém caracteres, o
heurístico calcula componentes conectados em cada candidato. São avaliados:

- quantidade de componentes;
- altura e largura relativas;
- razão e preenchimento de cada componente;
- consistência das alturas;
- alinhamento vertical adicional para placas horizontais.

Cinco componentes já saturam o benefício de contagem, enquanto quantidades
acima de dez são penalizadas. Isso favorece sequências plausíveis sem exigir
exatamente sete caracteres, o que seria frágil diante de borramento, oclusão ou
caracteres unidos.

Quando Otsu produz evidência fraca (`character_score < 0,45`), a limiarização
adaptativa é tentada. Ela custa processamento adicional, mas somente nos casos
difíceis. Manter o melhor dos dois resultados reduz dependência de iluminação
uniforme.

### Combinar evidências em vez de impor novas barreiras rígidas

O score final é uma soma ponderada:

```text
0,18 × proporção
+ 0,24 × área
+ 0,14 × posição
+ 0,34 × caracteres
+ 0,05 × contraste
+ 0,05 × preenchimento do contorno
```

Caracteres recebem o maior peso porque constituem a evidência mais específica
de uma placa. Área e proporção vêm em seguida por descreverem sua geometria. A
posição tem peso menor: placas costumam aparecer próximas à parte inferior e
ao centro do crop, mas uma regra dominante de posição falharia em veículos
parciais ou deslocados. Contraste e preenchimento atuam apenas como desempate,
pois variam muito com iluminação e segmentação.

Os termos de razão, área e posição usam curvas gaussianas, produzindo queda
gradual em torno dos alvos. Essa escolha explica parte do ganho de recall: uma
característica fora do ideal reduz o score, mas não elimina imediatamente um
candidato sustentado por outras evidências.

Os pesos são heurísticos e somam 1,0 para manter o resultado no intervalo
`0–1`; isso facilita interpretação e threshold, mas não transforma o score em
probabilidade calibrada.

### Retornar apenas o melhor candidato acima de 0,60

Depois do ranking, o benchmark aceita somente o candidato de maior score quando
ele alcança 0,60. Retornar uma única caixa limita falsos positivos e chamadas de
OCR por crop. O threshold 0,60 representa o ponto operacional escolhido para a
rodada: valores menores tenderiam a aumentar recall e custo de OCR; valores
maiores tenderiam a aumentar seletividade e falsos negativos.

Essa decisão deve ser interpretada junto às métricas observadas. Com 0,60, o
heurístico chegou a recall 0,6400 e precisão 0,6911. Portanto, o threshold não
elimina a necessidade de calibração por domínio; ele apenas explicita o
trade-off aplicado nesta avaliação.