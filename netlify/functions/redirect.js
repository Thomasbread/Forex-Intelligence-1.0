exports.handler = async function(event, context) {
  return {
    statusCode: 302,
    headers: {
      Location: 'https://forex-intelligence-app.streamlit.app'
    }
  };
};