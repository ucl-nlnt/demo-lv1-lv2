function createGradioAnimation() {
    dist = document.querySelector('#total-dist');
    degs = document.querySelector('#total-degs');
    breakdown = document.querySelector('#prompt');
    batt = document.querySelector('#batt');
    timeout_ms = 500
    setInterval(() => {
        fetch('http://127.0.0.1:8000/metadata')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                dist.innerHTML = `${data.total_distance_traveled}`
                degs.innerHTML = `${data.total_degrees_rotated}°`
                breakdown.innerHTML = `${data.prompt}`
                batt.innerHTML = `${data.battery_percentage}`
            })
            .catch(error => {
                console.error('There was a problem with the fetch operation:', error);
            });
        
    }, timeout_ms)
}