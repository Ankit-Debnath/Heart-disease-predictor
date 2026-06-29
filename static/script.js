const form = document.getElementById("predictionForm");
const result = document.getElementById("result");

form.addEventListener("submit", async function (e) {

    e.preventDefault();

    const formData = new FormData(form);

    const data = {};

    formData.forEach((value, key) => {

        data[key] = Number(value);

    });

    result.style.display = "block";
    result.className = "";
    result.innerHTML = "⏳ Predicting...";

    try {

        const response = await fetch("/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(data)

        });

        const prediction = await response.json();

        if (prediction.prediction === 1) {

            result.classList.add("danger");

            result.innerHTML = `
                ❤️ High Risk of Heart Disease
                <br><br>
                ${prediction.message}
            `;

        } else {

            result.classList.add("success");

            result.innerHTML = `
                ✅ Low Risk of Heart Disease
                <br><br>
                ${prediction.message}
            `;

        }

    }

    catch (error) {

        result.classList.add("danger");

        result.innerHTML = `
            ❌ Error connecting to server.
        `;

        console.error(error);

    }

});