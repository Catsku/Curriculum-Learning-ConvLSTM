import csv
#175438,175547
#Inicio 12/08/2019  11h10
indice_inicial1 = 135287
#fim 27/08/2019  12h
indice_final1 = 135603-1

with open("/timestampsRJ.csv", 'r') as arquivo:
    reader = csv.reader(arquivo)
    lista = list(reader)
    lista.pop(0)
    print(lista[indice_inicial1])
    print(lista[indice_final1])


