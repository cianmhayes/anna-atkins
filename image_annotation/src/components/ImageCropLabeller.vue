<template>
  <div>
    <div id="work-area" style="float: left">
      <div id="container">
        <img id="targetImage" ref="targetImage" :src="srcUrl" @load="updateCanvas"/>
        <canvas id="markupCanvas" ref="myCanvas" @click="onClick"></canvas>
      </div>
    </div>
    <div id="control-panel" style="float: right">
      <ol>
        <li v-for="(p, index) of points" :key="index" :class="getClass(index)">
          <b>{{ p.name }}</b> ({{ p.x }},{{ p.y }})
        </li>
      </ol>
      <p><b>Status:</b> {{ this.status }}</p>
      <button @click="submitLabels()">Submit</button>
      <button @click="initialize()">Reset</button>
    </div>
  </div>
</template>

<script>
export default {
  name: 'ImageCropLabeller',
  props: {
    imageId: String,
    srcUrl: String,
    regionNames: Array,
    autoIncrement: Boolean
  },
  data () {
    return {
      points: [],
      status: ''
    }
  },
  mounted () {
    window.addEventListener(
      'keypress',
      function (e) {
        const isNumber = /^[0-9]$/i.test(e.key)
        if (isNumber) {
          var index = parseInt(e.key) - 1
          if (index < 0) {
            index = 9
          }
          if (index >= this.points.length) {
            return
          }
          console.log(`Setting point ${index} to selected.`)
          for (var i = 0; i < this.points.length; i++) {
            this.points[i].selected = index === i
          }
        }
      }.bind(this)
    )
    this.initialize()
  },
  methods: {
    initialize () {
      this.points = []
      for (var i = 0; i < this.regionNames.length && i < 10; i++) {
        this.points.push({
          selected: false,
          name: this.regionNames[i],
          x: -1,
          y: -1
        })
      }
      this.points[0].selected = true
      this.status = 'Ready'
    },
    getClass (index) {
      if (index < this.points.length && this.points[index].selected) {
        return 'selectedItem'
      } else {
        return 'unselectedItem'
      }
    },
    onClick ({ clientX: x, clientY: y }) {
      var e = document.getElementById('targetImage')
      const eRect = e.getBoundingClientRect()
      for (var i = 0; i < this.points.length; i++) {
        if (this.points[i].selected) {
          this.points[i].x = x - eRect.left
          this.points[i].y = y - eRect.top
          if (this.autoIncrement === true && i < this.points.length - 1) {
            this.points[i].selected = false
            this.points[i + 1].selected = true
          }
          this.updateCanvas()
          return
        }
      }
    },
    updateCanvas () {
      const ti = this.$refs.targetImage
      const c = this.$refs.myCanvas
      const ctx = c.getContext('2d')
      c.width = ti.width
      c.height = ti.height
      this.points.forEach((p) => {
        if (p.x >= 0 && p.y >= 0) {
          ctx.beginPath()
          ctx.strokeStyle = 'red'
          ctx.lineWidth = '4'
          ctx.rect(p.x, p.y, 2, 2)
          ctx.stroke()
        }
      })
      console.log(this.points)
    },
    submitLabels () {
      this.status = 'Submitting'
      var e = document.getElementById('targetImage')
      const eRect = e.getBoundingClientRect()
      var requests = []
      for (var i = 0; i < this.points.length; i++) {
        const r = fetch(
          'https://cianh-anna-atkins.azurewebsites.net/api/images/' +
            this.imageId +
            '/annotations?code=y7NaJsT12QWx52HaWzNWeECLyusO8Dq/VbmRkSBvAaXV/v34Y9vW1A==',
          {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            mode: 'cors',
            body: JSON.stringify({
              image_id: this.imageId,
              annotation_name: this.points[i].name,
              x: (this.points[i].x * 1.0) / eRect.width,
              y: (this.points[i].y * 1.0) / eRect.height
            })
          }
        )
          .then((response) => {
            console.log(response.data)
          })
          .catch((error) => {
            alert(error)
          })
        requests.push(r)
      }
      const allRequests = Promise.all(requests)
      allRequests.then((response) => (this.status = 'Done'))
    }
  },
  watch: {
    points: function () {
      this.updateCanvas()
    }
  }
}
</script>

<style scoped>
#container {
  display: inline-block;
  position: relative;
}

#targetImage {
  position: absolute;
  z-index: 1;
}

#markupCanvas {
  position: relative;
  z-index: 20;
}

.selectedItem {
  background-color: #42b983;
}

.unselectedItem {
}
</style>
