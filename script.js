let detect = async () => {
  // Load model
  await faceapi.nets.ssdMobilenetv1.loadFromUri("/facedetect/models");
  await faceapi.nets.faceRecognitionNet.loadFromUri("/facedetect/models");
  await faceapi.nets.faceLandmark68Net.loadFromUri("/facedetect/models");
  
  const input = document.getElementById("myImg");

  let fullFaceDescriptions = await faceapi
    .detectAllFaces(input, new faceapi.SsdMobilenetv1Options())
    .withFaceLandmarks()
    .withFaceDescriptors();
 
  const displaySize = { width: input.width, height: input.height };
  const canvas = document.getElementById("myCanvas");
  faceapi.matchDimensions(canvas, displaySize);

  fullFaceDescriptions = faceapi.resizeResults(fullFaceDescriptions, displaySize);
  //console.log(fullFaceDescriptions);
  
  const labeledFaceDescriptors = await detectAllLabeledFaces();
  
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.7);
  //console.log(faceMatcher.toJSON());
  const final_results = fullFaceDescriptions.map(fd => faceMatcher.findBestMatch(fd.descriptor))

  if (fullFaceDescriptions) {
    final_results.forEach((bestMatch, i) => {
     const box = fullFaceDescriptions[i].detection.box
     const text = bestMatch.toString()
     const drawBox = new faceapi.draw.DrawBox(box, { label: text })
     drawBox.draw(canvas)
    })
  }
}

async function detectAllLabeledFaces() {
  const labels = ["Nancy", "Yeonwoo"];
  return Promise.all(
    labels.map(async label => {
      const descriptions = [];
      for (let i = 1; i <= 2; i++) {
        const img = await faceapi.fetchImage(
          `${window.location.href}data/${label}/${i}.jpg`
        );
        const detection = await faceapi
          .detectSingleFace(img)
          .withFaceLandmarks()
          .withFaceDescriptor();
        descriptions.push(detection.descriptor);
      }
      return new faceapi.LabeledFaceDescriptors(label, descriptions);
    })
  );
}

function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();

            reader.onload = function (e) {
                const input = document.getElementById("myImg").src = e.target.result;
                detect();
            };
            reader.readAsDataURL(input.files[0]);
        }
}
