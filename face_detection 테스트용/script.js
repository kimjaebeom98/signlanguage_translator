const video = document.getElementById('video')

// 모델 로드를 끝 마치면 startVideo 함수 실행 
Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
  faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
  faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
  faceapi.nets.faceExpressionNet.loadFromUri('/models'),
  faceapi.nets.ssdMobilenetv1.loadFromUri('./models')
]).then(startVideo)

// 유저의 카메라 권한을 얻기 위한 코드
function startVideo() {
  navigator.mediaDevices
    .getUserMedia({ video: true })
    .then(function (stream) {
      video.srcObject = stream;
    })
    .catch(function (err) {
      console.log(err);
    });
}


video.addEventListener('play', () => {
  // canvas를 초기화 함
  const canvas = faceapi.createCanvasFromMedia(video)
  document.body.append(canvas) 
  const displaySize = { width: video.width, height: video.height }
  faceapi.matchDimensions(canvas, displaySize)
  // 100ms 마다 화면에 video frame이 표시 됨
  setInterval(async () => {
    // video에서 얼굴을 식별
    const detections = await faceapi.detectAllFaces(video).withFaceLandmarks().withFaceDescriptors()
    const resizedDetections = faceapi.resizeResults(detections, displaySize)
    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)
    // video에서 얼굴 좌표에 box를 그림
    resizedDetections.forEach(detection => {
      const box = detection.detection.box
      const drawBox = new faceapi.draw.DrawBox(box, {label: 'Face'})
      drawBox.draw(canvas)
      // box의 좌표 값과 너비, 길이를 콘솔창에 출력
      console.log(box.x, box.y, box.width, box.height)
      if(box.x>=210 && box.x<=250 && box.y>=30 && box.y <=45){
        if(box.x+box.width>=370 && box.x+box.width <=410 && box.y+box.height >=180 && box.y+box.height <= 195 )
        {
            //소켓 이벤트를 방생시켜라 햇는데 이것보다 javascript코드에서 건너 뛰면 되지않을까?
        }

  })
  }, 100)
})