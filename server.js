// server.js
const express = require('express'); // loads Express
const app = express(); // creates an "app" (your server)

app.get('/', (req, res) => { // "/" means home page
  res.send('Hello, World!');
});

const PORT = 3000;
app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});