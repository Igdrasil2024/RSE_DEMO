function searchSite() {
    const input = document.getElementById("searchInput").value;
    const result = document.getElementById("result");

    if (!input) {
        alert("Veuillez entrer un nom ou une URL");
        return;
    }

    // Simulation d’un score (prototype)
    const score = (Math.random() * 10).toFixed(1);

    let label = "";
    let colorClass = "";

    if (score >= 8) {
        label = "?? Faible invasivité";
        colorClass = "green";
    } else if (score >= 5) {
        label = "?? Transparence moyenne";
        colorClass = "orange";
    } else {
        label = "?? Risque élevé";
        colorClass = "red";
    }

    document.getElementById("siteName").innerText = input;
    document.getElementById("scoreValue").innerText = `${score} / 10`;
    document.getElementById("scoreLabel").innerText = ` — ${label}`;

    const scoreDiv = document.querySelector(".score");
    scoreDiv.className = `score ${colorClass}`;

    result.classList.remove("hidden");
}
