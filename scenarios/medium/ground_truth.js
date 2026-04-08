async function getUser(userId, includeDetails = false) {
  try {
    const url = new URL(`https://api.example.com/users/${userId}`);
    url.searchParams.set("details", includeDetails ? "1" : "0");

    const response = await fetch(url.toString(), {
      headers: {
        Accept: "application/json",
        "X-Request-Source": "openenv"
      }
    });

    if (!response.ok) {
      throw new Error(`Failed: ${response.status}`);
    }

    const data = await response.json();
    data.source = "api";
    return data;
  } catch (error) {
    throw error;
  }
}