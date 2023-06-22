# Dados do Projeto

Este projeto contém diversos conjuntos de dados relacionados a check-ins locais e categorias gerais. Abaixo está uma descrição dos diretórios e arquivos contidos neste repositório.

## Diretório /data/basic

Este diretório contém os dados básicos necessários para a geração dos check-ins locais. Os arquivos disponíveis são:

- **gowalla_category**: Este arquivo contém as categorias e suas subcategorias. É usado para mapear as subcategorias para as categorias mais gerais.
- **gowalla_checkins.csv**: Este arquivo contém todos os check-ins sem nenhum preprocessamento.
- **places**: Este arquivo contém os dados dos locais, incluindo sua posição geográfica e subcategoria.

## Diretório /data/checkins_local

Este diretório contém um arquivo de check-in para cada estado dos Estados Unidos da América (EUA). Cada arquivo contém os check-ins específicos daquele estado, excluindo aqueles com categoryid igual a -1.

## Diretório /data/general

Este diretório contém arquivos gerados durante o preprocessamento dos dados. Os arquivos disponíveis são:

- **category_names**: Este arquivo contém o mapeamento de categoryid para o nome da categoria. Utilize este arquivo para obter o nome de uma categoria com base no categoryid.
- **gowalla_checkins_no_category**: Este arquivo contém os check-ins para os quais não foi possível encontrar uma categoria geral correspondente.
- **gowalla_checkins_with_category**: Este arquivo contém os check-ins para os quais foi possível encontrar uma categoria geral correspondente.
- **gowalla_geo_checkins**: Este arquivo contém os dados dos check-ins juntamente com informações geográficas. Este arquivo não inclui check-ins com categoryid igual a -1.
