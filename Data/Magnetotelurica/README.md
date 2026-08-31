# Datos de magnetotelúrica – Volcán Cerro Machín

Este directorio contiene los datos magnetotelúricos (MT) procesados y utilizados en el estudio del área geotérmica del Volcán Cerro Machín, Colombia.

## Estaciones magnetotelúricas

Los archivos `Datos_MT_UIS.csv` y `Datos_MT_SGC.csv` contienen la información de localización geográfica de las estaciones magnetotelúricas utilizadas en este estudio.

Los datos integran información proveniente de las siguientes fuentes:

- **Estaciones UIS:** corresponden a las estaciones magnetotelúricas adquiridas por la Universidad Industrial de Santander (UIS) durante la campaña de campo realizada en el área del Volcán Cerro Machín.

- **Estaciones SGC:** corresponden a las estaciones magnetotelúricas adquiridas y procesadas por el Servicio Geológico Colombiano (SGC), reportadas en el informe técnico *Caracterización magnetotelúrica del área geotérmica de Cerro Machín* de Herrera (2020).

## Carpeta `edi_processed`

Esta carpeta contiene los sondeos magnetotelúricos procesados utilizados en el estudio. Incluye los sondeos del SGC y los sondeos adquiridos y procesados por la UIS, almacenados en formato EDI.

## Carpeta `edi_interpolated`

Esta carpeta contiene los sondeos magnetotelúricos de la UIS y del SGC seleccionados para trabajar dentro de un rango de frecuencias común. Los datos fueron procesados e interpolados considerando períodos de hasta \(10^{1}\) s, con el propósito de disponer de un conjunto de sondeos compatible para su posterior análisis, interpretación e inversión magnetotelúrica.

## Referencia

- Herrera, J. (2020). *Caracterización magnetotelúrica del área geotérmica de Cerro Machín*. Bogotá: Servicio Geológico Colombiano.