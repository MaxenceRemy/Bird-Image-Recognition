commande pour lancé l'api:

uvicorn main:app --reload 

ensuite pour s'identifier, écrire dans un autre terminal:

curl -X POST "http://127.0.0.1:8000/token" -d "username=alice&password=wonderland"


Deuxième commande pour accéder à la route racine avec le token JWT et la clé API:

curl -X GET "http://127.0.0.1:8000/" -H "Authorization: Bearer VOTRE_TOKEN_JWT" -H "api-key: 1234567asdfgh"

Remplacez VOTRE_TOKEN_JWT par le token que vous avez reçu de la première commande. Par exemple, si vous avez reçu ce token :
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhbGljZSIsImV4cCI6MTcyMDM1OTY3Mn0.OG97S7oSmWWbLno0eJIlsxkqsrZKReMwzszm8P_u_Iw

Alors la deuxième commande complète serait :
curl -X GET "http://127.0.0.1:8000/" -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhbGljZSIsImV4cCI6MTcyMDM1OTY3Mn0.OG97S7oSmWWbLno0eJIlsxkqsrZKReMwzszm8P_u_Iw" -H "api-key: 1234567asdfgh"
