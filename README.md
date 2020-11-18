# SOIMA

\par La implementación presente del SOIMA requiere de diversos procesos off-line para caracterizar y extraer la información auditiva y visual de contenido multimedia (véase \textit{Estímulos}). Los SOMs que componen la arquitectura, se entrenan con la misma tasa de aprendizaje. El tamaño del vecindario de activación directa de los SOMs disminuye linealmente; empieza con un tamaño igual al mapa y disminuye a lo largo del proceso de entrenamiento, hasta afectar únicamente a la \textit{BMU}.
En cada SOM se utiliza la distancia euclideana como función de activación directa entre el input y el mapa .Las neuronas ganadoras de los SOMs modales se eligen únicamente a través de la activación directa del input; la neurona ganadora del $SOM_{M}$ mediante la función de activación total.
\begin{equ}[H]
  \caption{Activación directa de las unidades de los SOMs}
  \begin{equation}
    DirA_{j}=\sqrt{\displaystyle\sum_{i=0}^{i=\textit{n}} (v_{i}-w_{ij})^{2}}
  \end{equation}
\end{equ}
Donde $DirA_{j}$ es la activación directa de cada neurona del SOM tras el procesamiento del input \textit{v$_{i}$}; \textit{w$_{ij}$} es el peso que existe entre la neurona y el actual input. La $BMU$ es la neurona con el menor valor de \textit{DirA$_{j}$}.


\begin{equ}[H]
  \caption{Activación indirecta de las unidades de los SOMs}
  \begin{equation}
    IndA_{j}=vw_{ij}+sw_{ij}
  \end{equation}
\end{equ}
Donde $IndA_{j}$ es la activación indirecta de cada neurona como resultado de la asociación hebbiana entre los SOMs unimodales y el SOM$_{M}$; la medida representa las predicciones de activación del SOM$_{M}$ con base en el aprendizaje de las coocurrencias entre el SOM modal y el SOM$_{M}$; $vw_{ij}$ es el peso hebbiano entre la unidad $i$ del mapa multimodal con respecto de la unidad $j$ del mapa visual;$sw_{ij}$ es el peso hebbiano entre la unidad $i$ del mapa multimodal con respecto de la unidad $j$ del mapa auditivo.
\begin{equ}[H]
  \caption{Activación total de las unidades del SOM$_{M}$}
  \begin{equation}
    TotA_{j}^{n}=1-(MinMax[DirA_{j}^{n}])+(MinMax[IndA_{j}^{n}])
  \end{equation}
\end{equ}
Donde $TotA_{j}^{n}$ es la activación total de cada neurona $j$ del SOM$_{M}$; la medida representa la integración del grado de activación directa e indirecta y ocasiona un reajuste de la $BMU$ de cada respectiva vía hacía una que reduzca el error de manera parsimoniosa. La $BMU_{MT}$ es la neurona con mayor valor de $TotA_{j}$
\begin{equ}[H]
  \caption{Actualización de los pesos de los SOMs vía $DirA$}
  \begin{equation}
    \Delta w_{ij}= (\alpha\lambda)(v_{i}-w_{ij})
  \end{equation}
\end{equ}
Donde $\Delta$ $w_{ij}$ es el incremento en los pesos de cada neurona $ij$ del mapa, $w_{ij}$ es el peso original $ij$ del SOM, $v_{i}$ es el input, $\alpha$ es la tasa de aprendizaje $(0.3)$ y $\lambda$ es la proporción de ciclos de entrenamiento restantes. Cuando una neurona se encuentra fuera del vecindario de activación de la $BMU$ por vía directa, su incremento se iguala a 0.
\begin{equ}[H]
  \caption{Actualización de los pesos de asociación hebbiana}
  \begin{equation}
    \Delta w_{ij}= (\alpha\lambda)DirA_{i}DirA_{j}
  \end{equation}
\end{equ}
Donde $\Delta$ $w_{ij}$ es el incremento de los pesos en la asociación entre la neurona $i$ de un mapa y la neurona $j$ del otro, $\alpha$ es la tasa de aprendizaje $(0.3)$ y $\lambda$ es la proporción de ciclos de entrenamiento restantes, $DirA_{i}$ es la activación directa del elemento $i$ de un mapa, $DirA_{j}$ es la activación directa del elemento $j$ del mapa restante. Únicamente se actualiza el peso entre las neuronas $BMU_{i}$ y $BMU_{j}$. Los pesos de associación hebbiana se inicializan en cero para comenzar el entrenamiento del SOIMA.
