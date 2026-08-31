# Datos de gravimetría – Volcán Cerro Machín

Este directorio contiene los datos de gravimetría procesados y utilizados en el estudio del área geotérmica del Volcán Cerro Machín, Colombia.

## Archivos CSV

Los archivos `.csv` contienen la información de la localización geográfica de la red de estaciones de gravimetría procesadas y utilizadas en este estudio, junto con los valores gravimétricos, las correcciones aplicadas durante el procesamiento y las anomalías calculadas.

Los datos integran información proveniente de las siguientes fuentes:

- **Estaciones UIS:** corresponden a las estaciones de gravimetría adquiridas por la Universidad Industrial de Santander (UIS) durante la campaña de campo realizada en el área del Volcán Cerro Machín.

- **Estaciones SGC:** corresponden a las estaciones de gravimetría reportadas por el Servicio Geológico Colombiano (SGC) en el informe técnico *Gravimetría y magnetometría del área geotérmica del volcán Cerro Machín* de Beltrán (2020).

- **Archivos con la denominación `Paper`:** además de las estaciones UIS y SGC, incorporan las estaciones de gravimetría reportadas en el artículo científico *Gravity Studies at the Cerro Machín Volcano, Colombia* de Pedraza et al. (2022).

Los archivos CSV incluyen las coordenadas y elevación de las estaciones, los valores de gravedad, las diferentes correcciones aplicadas durante el procesamiento y las anomalías gravimétricas calculadas.

## Archivos NPY

Los archivos `.npy` contienen la información procesada utilizada para la inversión gravimétrica. Estos archivos incluyen las coordenadas espaciales X, Y y Z de las estaciones y los valores de la anomalía gravimétrica residual calculada.

Los archivos con `Paper` en su nombre corresponden a la base de datos que incorpora adicionalmente las estaciones reportadas por Pedraza et al. (2022).

## Referencias

- Beltrán (2020). *Gravimetría y magnetometría del área geotérmica del volcán Cerro Machín*. Servicio Geológico Colombiano (SGC).

- Pedraza et al. (2022). *Gravity Studies at the Cerro Machín Volcano, Colombia*.