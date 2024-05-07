function createGradioAnimation() {
    dist = document.querySelector('#total-dist');
    degs = document.querySelector('#total-degs');
    let total_distance_traveled = 0
    let total_degrees_rotated = 0
    setInterval(()=>{
        total_distance_traveled += 1
        total_degrees_rotated += 2
        dist.innerHTML = `${total_distance_traveled}`
        degs.innerHTML = `${total_degrees_rotated}°`
    }, 500)
}