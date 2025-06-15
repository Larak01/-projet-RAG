import openai

# Remplace ces variables par tes vraies infos Azure
openai.api_type = "azure"
openai.api_key = "sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXX"
openai.api_base = "https://TON-INSTANCE.openai.azure.com/"
openai.api_version = "2023-12-01-preview"

deployment_name = "gpt-35-turbo"  # nom du déploiement tel qu’il apparaît sur Azure

try:
    response = openai.ChatCompletion.create(
        engine=deployment_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ]
    )
    print("✅ Réponse reçue depuis Azure OpenAI :")
    print(response["choices"][0]["message"]["content"])
except Exception as e:
    print("❌ Erreur de connexion ou de configuration :")
    print(e)
