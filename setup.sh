mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"abiodun.msulaiman@gmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "
[theme]\n\
primaryColor = '#FF4B4B'\n\
backgroundColor = '#FFFFFF'
secondaryBackgroundColor = '#F0F2F6'
textColor = '#31333F'
font = 'sans serif'
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml